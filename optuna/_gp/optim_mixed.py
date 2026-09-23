from __future__ import annotations

import math
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna._gp.scipy_blas_thread_patch import single_blas_thread_if_scipy_v1_15_or_newer
from optuna.logging import get_logger


if TYPE_CHECKING:
    import scipy.optimize as so
    import torch

    from optuna._gp import batched_lbfgsb
    from optuna._gp.acqf import BaseAcquisitionFunc
else:
    from optuna import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")
    batched_lbfgsb = _LazyImport("optuna._gp.batched_lbfgsb")


_logger = get_logger(__name__)


class _AcqfEvalBudget:
    """Shared per-suggestion budget of surrogate+acquisition point evaluations."""

    def __init__(self, max_evals: int) -> None:
        self.max_evals = max_evals
        self.n_used = 0
        self.best_x: np.ndarray | None = None
        self.best_fval: float = -np.inf

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.n_used)

    def record(self, xs: np.ndarray, fvals: np.ndarray) -> None:
        xs_arr = np.atleast_2d(np.asarray(xs, dtype=float))
        fvals_arr = np.atleast_1d(np.asarray(fvals, dtype=float)).reshape(-1)
        if xs_arr.shape[0] != fvals_arr.shape[0]:
            raise ValueError(
                f"xs and fvals length mismatch: {xs_arr.shape[0]} vs {fvals_arr.shape[0]}."
            )
        self.n_used += int(fvals_arr.shape[0])
        finite = np.isfinite(fvals_arr)
        if not np.any(finite):
            return
        idx = int(np.argmax(np.where(finite, fvals_arr, -np.inf)))
        if fvals_arr[idx] > self.best_fval:
            self.best_fval = float(fvals_arr[idx])
            self.best_x = xs_arr[idx].copy()


class _BudgetedAcquisitionFunc:
    """Wraps an acquisition function and counts every evaluated design.

    ``eval_acqf_no_grad`` never exceeds the remaining budget: unevaluated
    points receive ``-inf``. ``eval_acqf`` (used with autograd) requires the
    caller to request at most ``remaining`` points so shapes stay consistent.
    """

    def __init__(self, acqf: BaseAcquisitionFunc, budget: _AcqfEvalBudget) -> None:
        self._acqf = acqf
        self._budget = budget
        self.length_scales = acqf.length_scales
        self.search_space = acqf.search_space

    @property
    def remaining(self) -> int:
        return self._budget.remaining

    @property
    def best_x(self) -> np.ndarray | None:
        return self._budget.best_x

    @property
    def best_fval(self) -> float:
        return self._budget.best_fval

    @property
    def n_used(self) -> int:
        return self._budget.n_used

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        n_points = 1 if x.ndim == 1 else int(x.shape[0])
        if n_points > self._budget.remaining:
            raise RuntimeError(
                f"Requested {n_points} acquisition evaluation(s) with only "
                f"{self._budget.remaining} remaining in the shared budget."
            )
        result = self._acqf.eval_acqf(x)
        self._budget.record(x.detach().cpu().numpy(), result.detach().cpu().numpy())
        return result

    def eval_acqf_no_grad(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        was_1d = x_arr.ndim == 1
        if was_1d:
            x_arr = x_arr[None, :]
        n_points = int(x_arr.shape[0])
        remaining = self._budget.remaining
        if remaining <= 0 or n_points == 0:
            out = np.full(n_points, -np.inf)
            return out[0] if was_1d else out
        n_eval = min(n_points, remaining)
        vals_eval = np.atleast_1d(
            np.asarray(self._acqf.eval_acqf_no_grad(x_arr[:n_eval]), dtype=float)
        ).reshape(-1)
        self._budget.record(x_arr[:n_eval], vals_eval)
        if n_eval == n_points:
            return vals_eval[0] if was_1d else vals_eval
        out = np.full(n_points, -np.inf)
        out[:n_eval] = vals_eval
        return out[0] if was_1d else out

    def eval_acqf_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self._acqf.eval_acqf_with_grad(x)


def _is_budgeted(acqf: BaseAcquisitionFunc) -> bool:
    return isinstance(acqf, _BudgetedAcquisitionFunc)


def _sample_acqf_params(
    acqf: BaseAcquisitionFunc, n: int, rng: np.random.RandomState
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, acqf.search_space.dim))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*balance properties of Sobol.*",
            category=UserWarning,
        )
        return acqf.search_space.sample_normalized_params(n, rng=rng)


def _gradient_ascent_batched(
    acqf: BaseAcquisitionFunc,
    initial_params_batched: np.ndarray,
    initial_fvals: np.ndarray,
    continuous_indices: np.ndarray,
    lengthscales: np.ndarray,
    tol: float,
    max_evals: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function optimizes the acquisition function using preconditioning.
    Preconditioning equalizes the variances caused by each parameter and
    speeds up the convergence.

    In Optuna, acquisition functions use Matern 5/2 kernel, which is a function of `x / l`
    where `x` is `normalized_params` and `l` is the corresponding lengthscales.
    Then acquisition functions are a function of `x / l`, i.e. `f(x / l)`.
    As `l` has different values for each param, it makes the function ill-conditioned.
    By transforming `x / l` to `zl / l = z`, the function becomes `f(z)` and has
    equal variances w.r.t. `z`.
    So optimization w.r.t. `z` instead of `x` is the preconditioning here and
    speeds up the convergence.
    As the domain of `x` is [0, 1], that of `z` becomes [0, 1/l].
    """
    assert initial_params_batched.ndim == 2
    if len(continuous_indices) == 0:
        return initial_params_batched, initial_fvals, np.zeros(len(initial_fvals), dtype=bool)
    if _is_budgeted(acqf) and acqf.remaining <= 0:  # type: ignore[union-attr]
        return (
            initial_params_batched,
            initial_fvals,
            np.zeros(len(initial_fvals), dtype=bool),
        )

    last_neg_fval: dict[int, float] = {}

    def negative_acqf_with_grad(
        scaled_x: np.ndarray, fixed_params: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        next_params = np.array(fixed_params)  # (B, dim)
        # Scale back to the original domain, i.e. [0, 1], from [0, 1/s].
        assert scaled_x.ndim == 2 and next_params.ndim == 2
        next_params[:, continuous_indices] = scaled_x * lengthscales
        if not _is_budgeted(acqf):
            # NOTE(Kaichi-Irie): If fvals.numel() > 1, backward() cannot be computed, so we sum up.
            x_tensor = torch.from_numpy(next_params).requires_grad_(True)
            neg_fvals = -acqf.eval_acqf(x_tensor)
            neg_fvals.sum().backward()  # type: ignore[no-untyped-call]
            grads = x_tensor.grad.detach().numpy()  # type: ignore[union-attr]
            neg_fvals_ = np.atleast_1d(neg_fvals.detach().numpy())
            # Flip sign because scipy minimizes functions.
            # Let the scaled acqf be g(x) and the acqf be f(sx), then dg/dx = df/dx * s.
            return neg_fvals_, grads[:, continuous_indices] * lengthscales

        batch_size = next_params.shape[0]
        n_eval = min(batch_size, acqf.remaining)  # type: ignore[union-attr]
        if n_eval <= 0:
            neg_fvals = np.fromiter(
                (last_neg_fval.get(id(fixed_params[i]), np.inf) for i in range(batch_size)),
                dtype=float,
                count=batch_size,
            )
            return neg_fvals, np.zeros((batch_size, len(continuous_indices)))

        x_eval = np.ascontiguousarray(next_params[:n_eval])
        x_tensor = torch.from_numpy(x_eval).requires_grad_(True)
        neg_fvals_t = -acqf.eval_acqf(x_tensor)
        neg_fvals_t.sum().backward()  # type: ignore[no-untyped-call]
        grads_eval = x_tensor.grad.detach().numpy()  # type: ignore[union-attr]
        neg_eval = np.atleast_1d(neg_fvals_t.detach().numpy())
        neg_fvals = np.empty(batch_size)
        grads = np.zeros((batch_size, next_params.shape[1]))
        neg_fvals[:n_eval] = neg_eval
        grads[:n_eval] = grads_eval
        for i in range(n_eval):
            last_neg_fval[id(fixed_params[i])] = float(neg_fvals[i])
        for i in range(n_eval, batch_size):
            # Zero grad makes L-BFGS-B treat this start as converged without a new eval.
            neg_fvals[i] = last_neg_fval.get(id(fixed_params[i]), np.inf)
        return neg_fvals, grads[:, continuous_indices] * lengthscales

    lbfgs_max_evals = max_evals
    if _is_budgeted(acqf):
        lbfgs_max_evals = max(acqf.remaining, 1)  # type: ignore[union-attr]
    lbfgsb_kwargs: dict = dict(
        func_and_grad=negative_acqf_with_grad,
        x0_batched=initial_params_batched[:, continuous_indices] / lengthscales,
        batched_args=([param for param in initial_params_batched.copy()],),
        bounds=[(0, 1 / s) for s in lengthscales],
        pgtol=math.sqrt(tol),
        max_iters=200,
    )
    if lbfgs_max_evals is not None:
        lbfgsb_kwargs["max_evals"] = lbfgs_max_evals
    with single_blas_thread_if_scipy_v1_15_or_newer():
        scaled_cont_xs_opt, neg_fvals_opt, n_iterations = batched_lbfgsb.batched_lbfgsb(
            **lbfgsb_kwargs
        )

    xs_opt = initial_params_batched.copy()
    xs_opt[:, continuous_indices] = scaled_cont_xs_opt * lengthscales
    # If any parameter is updated, return the updated parameters and values.
    # Otherwise, return the initial ones.
    fvals_opt = -neg_fvals_opt
    is_updated_batch = (fvals_opt > initial_fvals) & (n_iterations > 0)

    return (
        np.where(is_updated_batch[:, None], xs_opt, initial_params_batched),
        np.where(is_updated_batch, fvals_opt, initial_fvals),
        is_updated_batch,
    )


def _exhaustive_search(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    if len(choices) == 1:
        # Do not optimize anything when there's only one choice.
        return initial_params, initial_fval, False

    choices_except_current = choices[choices != initial_params[param_idx]]

    all_params = np.repeat(initial_params[None, :], len(choices_except_current), axis=0)
    all_params[:, param_idx] = choices_except_current
    fvals = acqf.eval_acqf_no_grad(all_params)
    best_idx = np.argmax(fvals)

    if fvals[best_idx] > initial_fval:  # Improved.
        return all_params[best_idx, :], fvals[best_idx], True

    return initial_params, initial_fval, False  # No improvement.


def _discrete_line_search(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    grids: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float, bool]:
    if len(grids) == 1:
        # Do not optimize anything when there's only one choice.
        return initial_params, initial_fval, False

    def find_nearest_index(x: float) -> int:
        i = int(np.clip(np.searchsorted(grids, x), 1, len(grids) - 1))
        return i - 1 if abs(x - grids[i - 1]) < abs(x - grids[i]) else i

    current_choice_i = find_nearest_index(initial_params[param_idx])
    assert np.isclose(initial_params[param_idx], grids[current_choice_i])

    negative_fval_cache = {current_choice_i: -initial_fval}

    normalized_params = initial_params.copy()

    def negative_acqf_with_cache(i: int) -> float:
        # Function value at choices[i].
        cache_val = negative_fval_cache.get(i)
        if cache_val is not None:
            return cache_val
        normalized_params[param_idx] = grids[i]

        # Flip sign because scipy minimizes functions.
        negval = -float(acqf.eval_acqf_no_grad(normalized_params))
        negative_fval_cache[i] = negval
        return negval

    def interpolated_negative_acqf(x: float) -> float:
        if x < grids[0] or x > grids[-1]:
            return np.inf
        right = int(np.clip(np.searchsorted(grids, x), 1, len(grids) - 1))
        left = right - 1
        neg_acqf_left, neg_acqf_right = (
            negative_acqf_with_cache(left),
            negative_acqf_with_cache(right),
        )
        w_left = (grids[right] - x) / (grids[right] - grids[left])
        w_right = 1.0 - w_left
        return w_left * neg_acqf_left + w_right * neg_acqf_right

    EPS = 1e-12
    res = so.minimize_scalar(
        interpolated_negative_acqf,
        # The values of this bracket are (inf, -fval, inf).
        # This trivially satisfies the bracket condition if fval is finite.
        bracket=(grids[0] - EPS, grids[current_choice_i], grids[-1] + EPS),
        method="brent",
        tol=xtol,
    )
    opt_idx = find_nearest_index(res.x)
    fval_opt = -negative_acqf_with_cache(opt_idx)

    # We check both conditions because of numerical errors.
    if opt_idx != current_choice_i and fval_opt > initial_fval:
        normalized_params[param_idx] = grids[opt_idx]
        return normalized_params, fval_opt, True

    return initial_params, initial_fval, False  # No improvement.


def _local_search_discrete(
    acqf: BaseAcquisitionFunc,
    initial_params: np.ndarray,
    initial_fval: float,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, float, bool]:
    # If the number of possible parameter values is small, we just perform an exhaustive search.
    # This is faster and better than the line search.
    MAX_INT_EXHAUSTIVE_SEARCH_PARAMS = 16

    is_categorical = acqf.search_space.is_categorical[param_idx]
    if is_categorical or len(choices) <= MAX_INT_EXHAUSTIVE_SEARCH_PARAMS:
        return _exhaustive_search(acqf, initial_params, initial_fval, param_idx, choices)
    else:
        return _discrete_line_search(acqf, initial_params, initial_fval, param_idx, choices, xtol)


def _local_search_discrete_batched(
    acqf: BaseAcquisitionFunc,
    initial_params_batched: np.ndarray,
    initial_fvals: np.ndarray,
    param_idx: int,
    choices: np.ndarray,
    xtol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO(Kaichi-Irie): Actually make it batched.
    best_normalized_params_batched = initial_params_batched.copy()
    best_fvals = initial_fvals.copy()

    is_updated_batch = np.zeros(len(initial_fvals), dtype=bool)
    for batch, normalized_params in enumerate(initial_params_batched):
        best_normalized_params, best_fval, updated = _local_search_discrete(
            acqf, normalized_params, best_fvals[batch], param_idx, choices, xtol
        )
        best_normalized_params_batched[batch] = best_normalized_params
        best_fvals[batch] = best_fval
        is_updated_batch[batch] = updated

    return best_normalized_params_batched, best_fvals, is_updated_batch


def local_search_mixed_batched(
    acqf: BaseAcquisitionFunc,
    xs0: np.ndarray,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
    max_evals: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # This is a technique for speeding up optimization. We use an isotropic kernel, so scaling the
    # gradient will make the hessian better-conditioned.
    # NOTE: Ideally, separating lengthscales should be used for the constraint functions,
    # but for simplicity, the ones from the objective function are being reused.
    # TODO(kAIto47802): Think of a better way to handle this.
    lengthscales = acqf.length_scales[(cont_inds := acqf.search_space.continuous_indices)]
    discrete_indices = acqf.search_space.discrete_indices
    choices_of_discrete_params = acqf.search_space.get_choices_of_discrete_params()
    discrete_xtols = [
        # Terminate discrete optimizations once the change in x becomes smaller than this.
        # Basically, if the change is smaller than min(dx) / 4, it is useless to see more details.
        np.min(np.diff(choices), initial=np.inf) / 4
        for choices in choices_of_discrete_params
    ]
    if _is_budgeted(acqf):
        n_keep = min(len(xs0), acqf.remaining)  # type: ignore[union-attr]
        if n_keep == 0:
            return xs0, np.full(len(xs0), -np.inf)
        xs0 = xs0[:n_keep]

    best_fvals = acqf.eval_acqf_no_grad((best_xs := xs0.copy()))
    CONTINUOUS = -1
    last_changed_dims = np.full(len(best_xs), CONTINUOUS, dtype=int)
    remaining_inds = np.arange(len(best_xs))
    for _ in range(max_iter):
        if _is_budgeted(acqf) and acqf.remaining <= 0:  # type: ignore[union-attr]
            return best_xs, best_fvals
        ga_max_evals = max_evals
        if _is_budgeted(acqf):
            ga_max_evals = max(acqf.remaining, 1)  # type: ignore[union-attr]
        best_xs[remaining_inds], best_fvals[remaining_inds], updated = _gradient_ascent_batched(
            acqf,
            best_xs[remaining_inds],
            best_fvals[remaining_inds],
            cont_inds,
            lengthscales,
            tol,
            max_evals=ga_max_evals,
        )
        last_changed_dims = np.where(updated, CONTINUOUS, last_changed_dims)
        for i, choices, xtol in zip(discrete_indices, choices_of_discrete_params, discrete_xtols):
            last_changed_dims = last_changed_dims[~(is_converged := last_changed_dims == i)]
            remaining_inds = remaining_inds[~is_converged]
            if remaining_inds.size == 0:
                return best_xs, best_fvals
            if _is_budgeted(acqf) and acqf.remaining <= 0:  # type: ignore[union-attr]
                return best_xs, best_fvals
            best_xs[remaining_inds], best_fvals[remaining_inds], updated = (
                _local_search_discrete_batched(
                    acqf, best_xs[remaining_inds], best_fvals[remaining_inds], i, choices, xtol
                )
            )
            last_changed_dims = np.where(updated, i, last_changed_dims)

        # Parameters not changed from the beginning or last changed dimension is continuous.
        remaining_inds = remaining_inds[~(is_converged := last_changed_dims == CONTINUOUS)]
        last_changed_dims = last_changed_dims[~is_converged]
        if remaining_inds.size == 0:
            return best_xs, best_fvals
    else:
        _logger.warning("local_search_mixed: Local search did not converge.")
    return best_xs, best_fvals


def optimize_acqf_mixed(
    acqf: BaseAcquisitionFunc,
    *,
    warmstart_normalized_params_array: np.ndarray | None = None,
    n_preliminary_samples: int = 2048,
    n_local_search: int = 10,
    tol: float = 1e-4,
    rng: np.random.RandomState | None = None,
    local_search: bool = True,
    n_acqf_evaluations: int | None = None,
) -> tuple[np.ndarray, float]:
    rng = rng or np.random.RandomState()

    if n_acqf_evaluations is not None and n_acqf_evaluations <= 0:
        raise ValueError("n_acqf_evaluations must be a positive integer or None.")

    if warmstart_normalized_params_array is None:
        warmstart_normalized_params_array = np.empty((0, acqf.search_space.dim))

    assert len(warmstart_normalized_params_array) <= n_local_search - 1, (
        "We must choose at least 1 best sampled point + given_initial_xs as start points."
    )

    budgeted = n_acqf_evaluations is not None
    if budgeted:
        acqf = _BudgetedAcquisitionFunc(acqf, _AcqfEvalBudget(n_acqf_evaluations))

    if not local_search:
        n_samples = n_acqf_evaluations if budgeted else n_preliminary_samples
        assert n_samples is not None
        sampled_xs = _sample_acqf_params(acqf, n_samples, rng)
        f_vals = acqf.eval_acqf_no_grad(sampled_xs)
        assert isinstance(f_vals, np.ndarray)
        # Preserve the unbudgeted extra incumbent eval; a fixed budget is random designs only.
        if not budgeted and len(warmstart_normalized_params_array) > 0:
            warm_f_vals = acqf.eval_acqf_no_grad(warmstart_normalized_params_array)
            assert isinstance(warm_f_vals, np.ndarray)
            f_vals = np.concatenate([f_vals, warm_f_vals])
            sampled_xs = np.vstack([sampled_xs, warmstart_normalized_params_array])
        if budgeted:
            assert acqf.best_x is not None  # type: ignore[union-attr]
            return acqf.best_x, acqf.best_fval  # type: ignore[union-attr, return-value]
        best_idx = np.argmax(f_vals).item()
        return sampled_xs[best_idx], f_vals[best_idx]

    n_samples = n_preliminary_samples
    if budgeted:
        assert n_acqf_evaluations is not None
        n_samples = min(n_preliminary_samples, n_acqf_evaluations)
    sampled_xs = _sample_acqf_params(acqf, n_samples, rng)
    f_vals = acqf.eval_acqf_no_grad(sampled_xs)
    assert isinstance(f_vals, np.ndarray)

    if budgeted and (acqf.remaining <= 0 or len(sampled_xs) == 0):  # type: ignore[union-attr]
        assert acqf.best_x is not None  # type: ignore[union-attr]
        return acqf.best_x, acqf.best_fval  # type: ignore[union-attr, return-value]

    max_i = np.argmax(f_vals)

    # TODO(nabenabe): Benchmark the BoTorch roulette selection as well.
    # https://github.com/pytorch/botorch/blob/v0.14.0/botorch/optim/initializers.py#L942
    # We use a modified roulette wheel selection to pick the initial param for each local search.
    probs = np.exp(f_vals - f_vals[max_i])
    probs[max_i] = 0.0  # We already picked the best param, so remove it from roulette.
    probs /= probs.sum()
    n_non_zero_probs_improvement = int(np.count_nonzero(probs > 0.0))
    # n_additional_warmstart becomes smaller when study starts to converge.
    n_additional_warmstart = min(
        n_local_search - len(warmstart_normalized_params_array) - 1, n_non_zero_probs_improvement
    )
    if n_additional_warmstart == n_non_zero_probs_improvement:
        _logger.warning("Study already converged, so the number of local search is reduced.")
    chosen_idxs = np.array([max_i])
    if n_additional_warmstart > 0:
        additional_idxs = rng.choice(
            len(sampled_xs), size=n_additional_warmstart, replace=False, p=probs
        )
        chosen_idxs = np.append(chosen_idxs, additional_idxs)

    x_warmstarts = np.vstack([sampled_xs[chosen_idxs, :], warmstart_normalized_params_array])
    best_xs, best_fvals = local_search_mixed_batched(acqf, x_warmstarts, tol=tol)
    if budgeted:
        assert acqf.best_x is not None  # type: ignore[union-attr]
        return acqf.best_x, acqf.best_fval  # type: ignore[union-attr, return-value]
    best_idx = np.argmax(best_fvals).item()
    return best_xs[best_idx], best_fvals[best_idx]
