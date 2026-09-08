"""Optional compute orchestration. Import starts no worker or provider connection."""
from .models import ComputeRequest, CuboidParameters, ConvolutionParameters, LocalPolicy, ResourceRequirements

__all__ = ['ComputeClient', 'ComputeRequest', 'CuboidParameters', 'ConvolutionParameters', 'LocalPolicy', 'ResourceRequirements', 'SSHProfile', 'WorkerConfig', 'SlurmConfig']


def __getattr__(name):
    if name == 'ComputeClient':
        from .client import ComputeClient
        return ComputeClient
    if name in {'SSHProfile', 'WorkerConfig', 'SlurmConfig'}:
        from . import remote
        return getattr(remote, name)
    raise AttributeError(name)
