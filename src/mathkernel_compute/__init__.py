"""Optional compute orchestration. Import starts no worker or provider connection."""
from .models import ComputeRequest, CuboidParameters, ConvolutionParameters, LocalPolicy, ResourceRequirements

__all__ = ['ComputeClient', 'ComputeRequest', 'CuboidParameters', 'ConvolutionParameters', 'LocalPolicy', 'ResourceRequirements']


def __getattr__(name):
    if name == 'ComputeClient':
        from .client import ComputeClient
        return ComputeClient
    raise AttributeError(name)
