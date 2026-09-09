"""Explicit execution eligibility. Mathematical algorithms remain in domain modules."""
from __future__ import annotations
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys
from .models import RuntimeProfile
from .protocol import digest

from .models import OperationDescriptor, CuboidParameters, ConvolutionParameters


def descriptors():
    return (
        OperationDescriptor(operation='cuboid_sweep', engine='python', optional_engines=('cuda',), output_schema='mk.cuboid-pairs/1',
            arithmetic='exact integers; bound 2..2000', claims=('witnesses', 'complete_search'),
            verifier='pythagorean-identities-and-euclid', parameter_schema_digest=digest(CuboidParameters.model_json_schema())),
        OperationDescriptor(operation='signal_convolve', engine='scipy', output_schema='mk.real-convolution/1',
            arithmetic='binary64 real samples; at most 256 per input', claims=('numeric_convolution',),
            verifier='decimal-reference-convolution', parameter_schema_digest=digest(ConvolutionParameters.model_json_schema())),
    )


# Lazy descriptor construction keeps the optional canonicalizer out of base import.
class _Registry:
    def __getitem__(self, key):
        for item in descriptors():
            if item.operation == key:
                return item.model_dump(mode='json')
        raise KeyError(key)


OPERATIONS = _Registry()


def runtime_profile(*, gpu=False):
    # Identity includes installed project code, not a worker's claimed image hash.
    root = Path(__file__).resolve().parent.parent
    hasher = hashlib.sha256()
    for package in ('mathkernel', 'mathkernel_artifacts', 'mathkernel_compute'):
        for path in sorted((root / package).rglob('*.py')):
            hasher.update(path.relative_to(root).as_posix().encode())
            hasher.update(hashlib.sha256(path.read_bytes()).digest())
    from mathkernel._version import __version__
    dependencies = tuple(f'{name}=={importlib.metadata.version(name)}' for name in
                         ('pydantic', 'numpy', 'sympy', 'scipy', 'rfc8785'))
    details = {'source': hasher.hexdigest(), 'python': platform.python_version(),
               'platform': sys.platform, 'dependencies': dependencies, 'kernel': __version__}
    accelerator = None
    if gpu:
        import cupy
        from .models import AcceleratorProfile
        accelerator = AcceleratorProfile(cupy_version=cupy.__version__,
            cuda_runtime=cupy.cuda.runtime.runtimeGetVersion(), cuda_driver=cupy.cuda.runtime.driverGetVersion())
        details['accelerator'] = accelerator.model_dump(mode='json')
    return RuntimeProfile(accelerator=accelerator, digest=digest(details), python=details['python'],
                          platform=sys.platform, kernel_version=__version__, dependencies=dependencies)


def execute(request):
    """Return data only; legacy MathResult metadata does not cross the worker boundary."""
    if request.operation == 'cuboid_sweep':
        from mathkernel.cuboid import sweep_leg_pairs
        bound = int(request.parameters.bound)
        result = sweep_leg_pairs(bound, engine=request.parameters.engine, workers=1, max_hits=bound*(bound-1)//2)
        return tuple((str(a), str(b)) for a, bs in sorted(result['pairs'].items()) for b in bs)
    if request.operation == 'signal_convolve':
        import sympy as sp
        from mathkernel.signal_processing import DiscreteSignal, convolve
        p = request.parameters
        left = DiscreteSignal(samples=tuple(sp.Float(float(v), 17) for v in p.left), input_trust='numeric')
        right = DiscreteSignal(samples=tuple(sp.Float(float(v), 17) for v in p.right), input_trust='numeric')
        _, result = convolve(left, right, mode='numeric', max_work=65536)
        return tuple(format(float(sp.re(v)), '.17g') for v in result.samples)
    raise ValueError('OPERATION_NOT_REMOTE_ELIGIBLE')
