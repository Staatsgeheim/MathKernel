"""Optional compute orchestration. Import starts no worker or provider connection."""
from .models import (ComputeRequest, CuboidParameters, ConvolutionParameters, LocalPolicy,
                     ResourceRequirements, BudgetLimit, Money, ManagedQuote, ManagedResources)

__all__ = ['ComputeClient', 'ComputeRequest', 'CuboidParameters', 'ConvolutionParameters', 'LocalPolicy',
           'ResourceRequirements', 'SSHProfile', 'WorkerConfig', 'SlurmConfig', 'ModalProfile', 'RunpodProfile',
           'S3Storage', 'BudgetLimit', 'Money', 'ManagedQuote', 'ManagedResources', 'BatchRequest', 'BatchShard',
           'LambdaProfile', 'LambdaProvisioner', 'LambdaWatchdog']


def __getattr__(name):
    if name == 'LambdaProfile':
        from .lambda_models import LambdaProfile
        return LambdaProfile
    if name == 'LambdaProvisioner':
        from .provisioning import LambdaProvisioner
        return LambdaProvisioner
    if name == 'LambdaWatchdog':
        from .watchdog import LambdaWatchdog
        return LambdaWatchdog
    if name == 'ComputeClient':
        from .client import ComputeClient
        return ComputeClient
    if name in {'SSHProfile', 'WorkerConfig', 'SlurmConfig'}:
        from . import remote
        return getattr(remote, name)
    if name in {'ModalProfile', 'RunpodProfile', 'S3Storage'}:
        from . import managed
        return getattr(managed, name)
    if name in {'BatchRequest', 'BatchShard'}:
        from . import batch
        return getattr(batch, name)
    raise AttributeError(name)
