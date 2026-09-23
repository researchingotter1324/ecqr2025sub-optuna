from __future__ import annotations

import numpy as np
import optuna
import optuna._gp.acqf as acqf_module
import optuna._gp.gp as optuna_gp
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space
from optuna.samplers import GPSampler
import pytest


class _CountingLogEI(acqf_module.LogEI):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.n_point_evals = 0

    def eval_acqf(self, x):  # type: ignore[no-untyped-def]
        n_points = 1 if x.ndim == 1 else int(x.shape[0])
        self.n_point_evals += n_points
        return super().eval_acqf(x)


def _make_counting_logei(
    *, dim: int = 2, n_train: int = 8, seed: int = 0, integer: bool = False
) -> _CountingLogEI:
    rng = np.random.RandomState(seed)
    X = rng.rand(n_train, dim)
    Y = rng.randn(n_train)
    Y = (Y - Y.mean()) / (Y.std() + 1e-12)
    if integer:
        search_space = gp_search_space.SearchSpace(
            {
                "x": optuna.distributions.FloatDistribution(0.0, 1.0),
                "y": optuna.distributions.IntDistribution(0, 7),
            }
        )
        is_categorical = np.array([False, False])
    else:
        search_space = gp_search_space.SearchSpace(
            {f"x{i}": optuna.distributions.FloatDistribution(0.0, 1.0) for i in range(dim)}
        )
        is_categorical = np.zeros(dim, dtype=bool)
    gpr = optuna_gp.fit_kernel_params(
        X=X,
        Y=Y,
        is_categorical=is_categorical,
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=True,
    )
    return _CountingLogEI(gpr=gpr, search_space=search_space, threshold=float(Y.max()))


def test_non_local_uses_n_acqf_evaluations_not_preliminary() -> None:
    acqf = _make_counting_logei()
    n_acqf = 32
    x_opt, f_opt = optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=8,
        n_acqf_evaluations=n_acqf,
        local_search=False,
        rng=np.random.RandomState(42),
        warmstart_normalized_params_array=np.array([[0.1, 0.2]]),
    )
    assert acqf.n_point_evals == n_acqf
    assert x_opt.shape == (2,)
    assert np.isfinite(f_opt)


def test_non_local_unbudgeted_keeps_preliminary_and_incumbent() -> None:
    acqf = _make_counting_logei()
    n_prelim = 8
    x_opt, f_opt = optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=n_prelim,
        local_search=False,
        rng=np.random.RandomState(42),
        warmstart_normalized_params_array=np.array([[0.1, 0.2]]),
    )
    assert acqf.n_point_evals == n_prelim + 1
    assert x_opt.shape == (2,)
    assert np.isfinite(f_opt)


def test_local_never_exceeds_shared_budget() -> None:
    acqf = _make_counting_logei()
    n_prelim = 8
    n_acqf = 24
    x_opt, f_opt = optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=n_prelim,
        n_acqf_evaluations=n_acqf,
        local_search=True,
        rng=np.random.RandomState(42),
    )
    assert n_prelim <= acqf.n_point_evals <= n_acqf
    assert x_opt.shape == (2,)
    assert np.isfinite(f_opt)


def test_local_stops_when_budget_exhausted() -> None:
    acqf = _make_counting_logei()
    n_prelim = 16
    n_acqf = 18  # only 2 evals left for local search
    x_opt, f_opt = optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=n_prelim,
        n_acqf_evaluations=n_acqf,
        local_search=True,
        rng=np.random.RandomState(42),
    )
    assert acqf.n_point_evals == n_acqf
    assert x_opt.shape == (2,)
    assert np.isfinite(f_opt)


def test_local_mixed_discrete_respects_budget() -> None:
    acqf = _make_counting_logei(integer=True)
    n_prelim = 8
    n_acqf = 20
    x_opt, f_opt = optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=n_prelim,
        n_acqf_evaluations=n_acqf,
        local_search=True,
        rng=np.random.RandomState(0),
    )
    assert acqf.n_point_evals <= n_acqf
    assert x_opt.shape == (2,)
    assert np.isfinite(f_opt)


def test_local_unbudgeted_does_more_than_preliminary() -> None:
    acqf = _make_counting_logei()
    n_prelim = 8
    optim_mixed.optimize_acqf_mixed(
        acqf,
        n_preliminary_samples=n_prelim,
        local_search=True,
        rng=np.random.RandomState(42),
    )
    assert acqf.n_point_evals > n_prelim


def test_n_acqf_evaluations_must_be_positive() -> None:
    acqf = _make_counting_logei()
    with pytest.raises(ValueError):
        optim_mixed.optimize_acqf_mixed(
            acqf, n_acqf_evaluations=0, local_search=False, rng=np.random.RandomState(0)
        )


@pytest.mark.parametrize("local_search", [False, True])
def test_gpsampler_reference_configs_run(local_search: bool) -> None:
    n_acqf = 16
    n_prelim = 8
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        sampler = GPSampler(
            seed=42,
            n_startup_trials=0,
            deterministic_objective=True,
            local_search=local_search,
            n_preliminary_samples=n_prelim,
            n_acqf_evaluations=n_acqf,
        )
    assert sampler._n_acqf_evaluations == n_acqf
    assert sampler._n_preliminary_samples == n_prelim
    assert sampler._local_search is local_search
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x", -1.0, 1.0) ** 2, n_trials=3)
    assert len(study.trials) == 3
    assert study.best_trial.values is not None


def test_gpsampler_rejects_non_positive_budget() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        with pytest.raises(ValueError):
            GPSampler(n_acqf_evaluations=0)
