from __future__ import annotations
import numpy as np
import pytest
import sympy as sp
from mathkernel import MathKernel, Settings, TrustLevel
from mathkernel.signal_processing import DiscreteSignal, exact_dft


def create(k, typ, **definition):
    result = k.object_create(typ, definition)
    assert result.ok, result.errors
    return result.data['object_id']


@pytest.mark.parametrize('normalization', ['backward', 'forward', 'ortho'])
@pytest.mark.parametrize('samples', [[1], [1,2], [1,2,3,4], [0,1,0,-1]])
def test_exact_dft_matches_independent_definition_and_reconstructs(normalization, samples):
    k = MathKernel()
    src = create(k, 'Signal', samples=samples, sample_rate=8)
    before = k.object_get(src).model_dump(mode='json')
    out = k.apply(src, 'dft', {'normalization': normalization})
    assert out.ok, out.errors
    assert out.trust == TrustLevel.EXACT
    spectrum = k.math_objects[out.data['object_id']]['value']
    n = len(samples)
    scale = 1 if normalization == 'backward' else 1/n if normalization == 'forward' else 1/np.sqrt(n)
    expected = [scale*sum(v*np.exp(-2j*np.pi*j*i/n) for j,v in enumerate(samples)) for i in range(n)]
    np.testing.assert_allclose([complex(x.evalf()) for x in spectrum.bins], expected, atol=1e-13)
    restored = k.apply(out.data['object_id'], 'idft')
    assert restored.ok
    assert k.math_objects[restored.data['object_id']]['value'].samples == tuple(map(sp.Integer, samples))
    assert k.object_get(src).model_dump(mode='json') == before


def test_fast_fft_does_not_upgrade_decimal_or_exact_inputs():
    k = MathKernel()
    source = create(k, 'Signal', samples=[1,2,3,4])
    result = k.apply(source, 'dft', {'mode':'numeric'})
    assert result.trust == TrustLevel.NUMERIC
    assert result.data['details']['precision_bits'] == 53
    assert result.data['details']['verification_absolute_tolerance'] > 0
    assert result.data['provenance']['engine_versions']['numpy'] == np.__version__
    restored = k.apply(result.data['object_id'], 'idft', {'mode':'numeric'})
    assert restored.trust == TrustLevel.NUMERIC
    assert restored.evidence_bundle.numerical
    assert restored.data['provenance']['required_object_inputs'][0]['trust'] == 'numeric'


def test_high_precision_fft_preserves_more_than_float64_information():
    k = MathKernel()
    source = create(k, 'Signal', samples=['1+1/10^30', '-1'])
    out = k.apply(source, 'dft', {'mode':'numeric', 'precision':160})
    assert out.ok, out.errors
    assert out.trust == TrustLevel.NUMERIC_HIGH_PRECISION
    value = k.math_objects[out.data['object_id']]['value'].bins[0]
    assert abs(value-sp.Rational(1,10**30)) < sp.Rational(1,10**45)


def test_convolution_correlation_and_all_operand_ancestry():
    k = MathKernel()
    a = create(k, 'Signal', samples=[1,2,3], sample_rate=2, unit='V')
    b = create(k, 'Signal', samples=['0.5', 1], sample_rate=2, unit='A')
    for operation in ['convolution','correlation']:
        r = k.apply(a,operation,{'other_id':b})
        assert r.ok, r.errors
        assert r.trust == TrustLevel.NUMERIC
        child = k.object_get(r.data['object_id'])
        assert set(child.data['sources']) == {a,b}
        assert child.trust == TrustLevel.NUMERIC
        values = [float(x) for x in k.math_objects[r.data['object_id']]['value'].samples]
        expected = np.convolve([1,2,3],[.5,1]) if operation == 'convolution' else np.correlate([1,2,3],[.5,1],mode='full')
        np.testing.assert_allclose(values,expected)


def test_filter_recurrence_and_roundtrip_to_control():
    k=MathKernel()
    filt=create(k,'Filter',numerator=[1],denominator=[1,'-1/2'],sample_rate=2)
    signal=create(k,'Signal',samples=[1,0,0,0,0],sample_rate=2)
    out=k.apply(filt,'apply_signal',{'signal_id':signal})
    assert out.ok and out.data['value']==['1','1/2','1/4','1/8','1/16']
    tf=k.apply(filt,'to_transfer_function')
    restored=k.apply(tf.data['object_id'],'to_filter')
    again=k.apply(restored.data['object_id'],'apply_signal',{'signal_id':signal})
    assert again.data['value']==out.data['value']
    fast=k.apply(filt,'apply_signal',{'signal_id':signal,'mode':'numeric'})
    assert fast.ok and fast.trust==TrustLevel.NUMERIC
    assert fast.data['verification']['difference_equation_numeric']


@pytest.mark.parametrize('family',['butterworth','chebyshev1','chebyshev2','bessel','elliptic'])
def test_iir_design_uses_sos_and_keeps_design_claim_numeric(family):
    k=MathKernel()
    design=create(k,'FilterDesign',family=family,order=4,cutoff=[10],sample_rate=100)
    out=k.apply(design,'design',{'mode':'numeric'})
    assert out.ok,out.errors
    assert out.trust==TrustLevel.NUMERIC
    assert out.data['details']['specifications_certified'] is False
    filt=k.math_objects[out.data['object_id']]['value']
    assert filt.sos
    signal=create(k,'Signal',samples=[1]+[0]*99,sample_rate=100)
    response=k.apply(out.data['object_id'],'apply_signal',{'signal_id':signal,'mode':'numeric'})
    assert response.ok,response.errors
    assert response.data['verification']['difference_equation_numeric']
    assert any(e.method=='scipy_sosfilt' for e in response.evidence_bundle.computation)


def test_stft_resample_window_and_cross_spectrum_metadata():
    k=MathKernel()
    src=create(k,'Signal',samples=[1,2,3,4,5,6,7,8],sample_rate=8)
    w=k.apply(src,'window',{'kind':'hann'})
    assert w.ok and w.data['details']['periodic']
    stft=k.apply(src,'stft',{'size':4,'overlap':2,'mode':'numeric'})
    assert stft.ok,stft.errors
    assert len(stft.data['value'])==3
    assert stft.data['details']['padding']==0
    assert stft.data['details']['frame_starts']==['0','1/4','1/2']
    res=k.apply(src,'resample',{'up':2,'down':1,'mode':'numeric'})
    assert res.ok and len(res.data['value'])==16
    assert res.data['details']['aliasing_bound'] is None
    a=create(k,'Signal',samples=[1,2,3,4])
    cross=k.apply(a,'cross_spectrum',{'other_id':a})
    assert cross.ok and cross.trust==TrustLevel.EXACT
    restored=k.apply(cross.data['object_id'],'idft')
    assert restored.data['value']==['30','24','22','24']


def test_signal_limits_units_and_strict_integer_controls():
    k=MathKernel(Settings(max_signal_samples=8,max_exact_dft_size=2))
    a=create(k,'Signal',samples=[1,2,3,4],sample_rate=2)
    assert not k.apply(a,'dft').ok
    assert not k.object_create('Signal',{'samples':list(range(9))}).ok
    b=create(k,'Signal',samples=[1],sample_rate=3)
    assert not k.apply(a,'convolution',{'other_id':b}).ok
    assert not k.apply(a,'resample',{'mode':'numeric','up':1.5}).ok
    assert not k.apply(a,'resample').ok
    assert not k.object_create('Signal',{'samples':[1],'unit':'made_up_unit'}).ok
    assert not k.object_create('Signal',{'samples':['1/0']}).ok
    assert not k.object_create('Signal',{'samples':['__import__("os")']}).ok


def test_numeric_convolution_matches_direct_complex_definition():
    k=MathKernel()
    a=create(k,'Signal',samples=['complex(1,2)',2,-3])
    b=create(k,'Signal',samples=[1,'complex(0,1)'])
    r=k.apply(a,'convolution',{'other_id':b,'mode':'numeric'})
    assert r.ok,r.errors
    actual=k.math_objects[r.data['object_id']]['value'].samples
    np.testing.assert_allclose([complex(x) for x in actual],np.convolve([1+2j,2,-3],[1,1j]))


def test_exact_dft_16_algebraic_reconstruction_is_decided_exactly():
    from mathkernel.engineering import exact_zero
    samples=tuple(map(sp.Integer,range(16)))
    result=exact_dft(samples)
    recovered=exact_dft(result,inverse=True)
    assert all(exact_zero(x-y) is True for x,y in zip(samples,recovered))


def test_high_precision_roundtrip_retains_precision_ancestry():
    k=MathKernel();src=create(k,'Signal',samples=[1,2,3,4])
    forward=k.apply(src,'dft',{'mode':'numeric','precision':160})
    restored=k.apply(forward.data['object_id'],'idft',{'mode':'numeric','precision':160})
    assert restored.ok,restored.errors
    assert restored.trust==TrustLevel.NUMERIC_HIGH_PRECISION


def test_large_numeric_window_does_not_use_symbolic_path():
    k=MathKernel(Settings(max_exact_window_size=8))
    src=create(k,'Signal',samples=list(range(32)))
    assert not k.apply(src,'window').ok
    result=k.apply(src,'window',{'mode':'numeric'})
    assert result.ok,result.errors
    assert result.trust==TrustLevel.NUMERIC
    spectrum=k.apply(result.data['object_id'],'dft',{'mode':'numeric'})
    assert spectrum.data['details']['source_transformations']==['hann window; periodic=True']
