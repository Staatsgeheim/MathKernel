"""Public schema catalog generated from the same strict runtime contracts."""
from .models import (ComputeRequest, ComputePlan, AttemptRecord, RemoteResultEnvelope,
                     AuthorizationGrant, BudgetLimit, VerificationReport)
from .remote import SSHProfile, WorkerConfig, GatewayRequest
from .managed import ModalProfile, RunpodProfile
from .lambda_models import LambdaProfile, VMPlan, VMGrant, WatchdogTicket
from .batch import BatchRequest, BatchPlan
from .replay import Replay

CONTRACTS = (ComputeRequest, ComputePlan, AttemptRecord, RemoteResultEnvelope, AuthorizationGrant,
             BudgetLimit, VerificationReport, SSHProfile, WorkerConfig, GatewayRequest,
             ModalProfile, RunpodProfile, LambdaProfile, VMPlan, VMGrant, WatchdogTicket,
             BatchRequest, BatchPlan, Replay)


def schema_catalog():
    return {'schema': 'mk.schema-catalog/1', 'json_schema_dialect': 'https://json-schema.org/draft/2020-12/schema',
            'contracts': {c.__name__: c.model_json_schema() for c in CONTRACTS},
            'note': 'JSON Schema documents shape. Runtime also enforces bounded decoding, canonical hashes, ownership and host authority.'}
