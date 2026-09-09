from itertools import product

import numpy as np
import pytest

from mathkernel.robust_relation_inference import (
    crossfit_nuisance_projection, crossfit_residual_relations,
    gaussian_quadratic_test, prewhitened_long_run_covariance,
    quadratic_minimax_bounds, quadratic_moment_test, quadratic_u_test,
    relation_folds,
)


def test_u_statistic_matches_explicit_distinct_pairs():
    z = np.random.default_rng(2).normal(size=(13,4))
    a = np.array([.9,.7,.2,.1])
    explicit = sum(np.dot(z[i]/a,z[j]) for i in range(13) for j in range(13) if i!=j)/(13*12)
    assert quadratic_u_test(z,a)['statistic'] == pytest.approx(explicit)


def test_u_exact_binary_null_size_and_variance():
    # Exhaustive finite support, not a formula-mirroring Monte Carlo check.
    values = []
    rejects = []
    for sequence in product([-1.,1.],repeat=8):
        result = quadratic_u_test(np.array(sequence),[.3],alpha=.1)
        values.append(result['statistic']); rejects.append(result['reject'])
    assert np.mean(values)==pytest.approx(0,abs=1e-14)
    assert np.var(values)==pytest.approx(result['null_variance'])
    assert np.mean(rejects)<=.1


def test_gaussian_mixture_second_moment_lower_bound():
    rng=np.random.default_rng(45)
    a=np.array([.9,.3,.12]);epsilon=.2;n=60
    theta=epsilon/a/np.linalg.norm(1/a)
    delta=np.sqrt(n*a)*theta
    # Integrate the explicit sign-mixture likelihood under the null.
    z=rng.normal(size=(200000,3))
    likelihood=np.exp(-np.sum(delta**2)/2)*np.cosh(z*delta).prod(1)
    exact=np.cosh(delta**2).prod()
    assert np.mean(likelihood**2)==pytest.approx(exact,rel=.012)
    assert exact<=np.exp(n*n*epsilon**4/(2*np.sum(a**-2)))+1e-12


def test_isotropic_rate_has_sqrt_dimension_not_dimension():
    first=quadratic_minimax_bounds([.4]*4,.15)
    later=quadratic_minimax_bounds([.4]*64,.15)
    assert later.inverse_spectrum_l2/first.inverse_spectrum_l2==pytest.approx(4)
    assert later.gaussian_necessary_samples/first.gaussian_necessary_samples==pytest.approx(4,rel=.01)


def test_anisotropic_bounds_use_entire_spectrum():
    good=quadratic_minimax_bounds([1,1,.1],.2)
    bad=quadratic_minimax_bounds([.1,.1,.1],.2)
    assert good.weakest_information==bad.weakest_information
    assert good.gaussian_necessary_samples<bad.gaussian_necessary_samples
    assert good.gaussian_sufficient_samples<bad.gaussian_sufficient_samples


def test_bounds_bracket_and_detect_blindness():
    r=quadratic_minimax_bounds([.64,.64,.4096],.15,alternative_covariance_envelope=1.26)
    assert r.gaussian_necessary_samples<r.gaussian_sufficient_samples<r.iid_u_sufficient_samples
    blind=quadratic_minimax_bounds([.5,0],.15)
    assert blind.blind_direction and np.isinf(blind.gaussian_sufficient_samples)


@pytest.mark.parametrize('a,epsilon', [([-1],.1),([np.nan],.1),([],.1),([.5],0),([.5],np.inf)])
def test_bounds_reject_invalid(a,epsilon):
    with pytest.raises(ValueError):quadratic_minimax_bounds(a,epsilon)


def test_gaussian_test_invariant_to_permutation():
    a=np.array([.8,.3,.05]);z=np.array([3.,1.,5.])
    a1=gaussian_quadratic_test(z,a,100)
    a2=gaussian_quadratic_test(z[::-1],a[::-1],100)
    assert a1['statistic']==pytest.approx(a2['statistic'])
    assert a1['threshold']==pytest.approx(a2['threshold'])


def test_hac_recoloring_matches_multivariate_var_truth():
    pytest.importorskip("scipy", reason="Install mathkernel[test] for optional backend coverage")
    from scipy.signal import lfilter
    rng=np.random.default_rng(29)
    # A diagonal VAR in a rotated chart tests orientation of recoloring.
    q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    eigen=np.array([.85,.5,-.4])
    n=100000
    z=np.column_stack([lfilter([1],[1,-phi],rng.normal(size=n+200))[200:] for phi in eigen])@q.T
    result=prewhitened_long_run_covariance(z)
    true=q@np.diag(1/(1-eigen)**2)@q.T
    assert np.linalg.norm(result.covariance-true)/np.linalg.norm(true)<.05


def test_hac_manual_zero_bandwidth_matches_covariance():
    z=np.random.default_rng(31).normal(size=(400,3))
    r=prewhitened_long_run_covariance(z,prewhiten=False,bandwidth=0)
    assert np.allclose(r.covariance,np.cov(z,rowvar=False,bias=True))


def test_hac_fails_closed_on_degeneracy():
    with pytest.raises(ValueError):prewhitened_long_run_covariance(np.ones((100,2)))
    with pytest.raises(ValueError):prewhitened_long_run_covariance(np.arange(100)[:,None],bandwidth=-1)
    with pytest.raises(ValueError):quadratic_moment_test(np.ones((50,2)),covariance=np.diag([1.,0.]))


def test_group_folds_never_leak_groups():
    groups=np.repeat(np.arange(20),5)
    f=relation_folds(100,groups=groups)
    for g in np.unique(groups):assert len(np.unique(f[groups==g]))==1
    assert len(np.unique(f))==5


def test_projection_recovers_unknown_mixing_and_retains_mean_shift():
    rng=np.random.default_rng(7)
    u=rng.normal(size=(30000,2)); noise=rng.normal(size=(30000,1))
    t=2+u@np.array([[2.],[-1.]])+noise
    f=relation_folds(len(t))
    output=crossfit_nuisance_projection(t,u,f)
    assert abs(output.scores.mean()-2)<.02
    assert np.var(output.scores)==pytest.approx(1,rel=.025)
    transformed=crossfit_nuisance_projection(t,u@np.array([[2.,3.],[.5,2.]]),f)
    assert np.allclose(output.scores,transformed.scores,atol=1e-12)


def test_residual_learner_cannot_see_evaluation_labels_or_gap_neighbors():
    n=60;x=np.arange(n)[:,None].astype(float)
    folds=relation_folds(n,n_splits=3,contiguous=True)
    calls=[]
    def learner(train_x,train_y,test_x):
        calls.append((train_x[:,0],test_x[:,0]))
        assert np.min(np.abs(train_x[:,0,None]-test_x[:,0]))>3
        return np.zeros((len(test_x),train_y.shape[1]))
    output=crossfit_residual_relations(x,x,x,folds,learner=learner,gap=3)
    assert np.allclose(output.scores,x*x)
    assert len(calls)==6


def test_crossfitting_removes_measured_confounding():
    rng=np.random.default_rng(70);n=30000;x=rng.normal(size=(n,2))
    a=2*x[:,0]+x[:,1]+rng.normal(size=n)
    b=-x[:,0]+3*x[:,1]+rng.normal(size=n)
    output=crossfit_residual_relations(a,b,x,relation_folds(n))
    assert abs(output.scores.mean())<.03
    assert abs(np.mean(a*b))>.8


def test_nuisance_prediction_shape_and_fold_errors():
    x=np.arange(60).reshape(30,2);f=relation_folds(30)
    with pytest.raises(ValueError):
        crossfit_residual_relations(x,x,x,f,learner=lambda *args:np.nan)
    with pytest.raises(ValueError):crossfit_nuisance_projection(x,x,np.zeros(30,dtype=int))
    with pytest.raises(ValueError):relation_folds(30,groups=np.zeros(30))


def test_numerical_limit_is_not_reported_as_mathematical_impossibility():
    with pytest.raises(OverflowError,match='not an impossibility'):
        quadratic_minimax_bounds([1e-30],.1)
