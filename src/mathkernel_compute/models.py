"""Strict contracts. Nested request values are immutable; journal bytes are authoritative."""
from __future__ import annotations
from decimal import Decimal
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator, model_serializer
from .protocol import digest

Identifier = Annotated[str, Field(pattern=r'^[A-Za-z0-9_-]{1,128}$')]
Digest = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
IntegerText = Annotated[str, Field(pattern=r'^(0|[1-9][0-9]*)$', max_length=16)]
DecimalText = Annotated[str, Field(pattern=r'^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]{1,3})?$', max_length=64)]


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, validate_default=True)


class CuboidParameters(Contract):
    bound: IntegerText
    engine: Literal['python'] = 'python'

    @field_validator('bound')
    @classmethod
    def bounded(cls, v):
        if not 2 <= int(v) <= 2000:
            raise ValueError('bound must be between 2 and 2000 in this profile')
        return v


class ConvolutionParameters(Contract):
    left: tuple[DecimalText, ...] = Field(min_length=1, max_length=256)
    right: tuple[DecimalText, ...] = Field(min_length=1, max_length=256)
    engine: Literal['scipy'] = 'scipy'

    @field_validator('left', 'right')
    @classmethod
    def bounded(cls, values):
        for v in values:
            n = Decimal(v)
            if not n.is_finite() or abs(n) > Decimal('1e100') or (n != 0 and abs(n) < Decimal('1e-100')):
                raise ValueError('numeric samples must be finite and within the profile range')
        return values


class ResourceRequirements(Contract):
    execution_timeout_ms: int = Field(default=60000, ge=100, le=60000)
    verification_timeout_ms: int = Field(default=60000, ge=100, le=60000)
    max_output_bytes: int = Field(default=1_048_576, ge=4096, le=1_048_576)
    threads: Literal[1] = 1
    gpu_count: Literal[0] = 0

    @field_validator('threads', 'gpu_count', mode='before')
    @classmethod
    def strict_count(cls, value):
        if type(value) is not int:
            raise ValueError('resource counts require strict integers')
        return value


class ComputeRequest(Contract):
    operation: Literal['cuboid_sweep', 'signal_convolve']
    parameters: CuboidParameters | ConvolutionParameters
    operation_version: Literal['1'] = '1'
    target: Identifier = 'local-cpu'
    verification: Literal['local_required', 'inspect_only'] = 'local_required'
    required_claim: Literal['complete_search', 'witnesses', 'numeric_convolution'] = 'complete_search'
    resources: ResourceRequirements = Field(default_factory=ResourceRequirements)

    @model_validator(mode='after')
    def compatible(self):
        if self.operation == 'cuboid_sweep':
            if not isinstance(self.parameters, CuboidParameters) or self.required_claim == 'numeric_convolution':
                raise ValueError('cuboid_sweep requires integer parameters and a search claim')
        elif not isinstance(self.parameters, ConvolutionParameters) or self.required_claim != 'numeric_convolution':
            raise ValueError('signal_convolve requires numeric samples and numeric_convolution claim')
        return self


class TargetCapabilities(Contract):
    process_deadline: Literal['supported', 'unsupported']
    process_tree_cleanup: Literal['supported', 'unsupported']
    output_retention: Literal['controller-spool', 'remote-spool'] = 'controller-spool'
    network_isolation: Literal['unsupported'] = 'unsupported'
    hard_memory_limit: Literal['unsupported'] = 'unsupported'
    hard_cost_cap: Literal['unsupported'] = 'unsupported'
    source: Literal['local-runtime', 'operator-profile'] = 'local-runtime'


class ComputeTarget(Contract):
    target_id: Identifier = 'local-cpu'
    adapter: Literal['local', 'ssh', 'slurm'] = 'local'
    ownership: Literal['existing'] = 'existing'
    available: bool
    capabilities: TargetCapabilities
    reason: str


class InputBundle(Contract):
    bundle_schema: Literal['mk.bundle/1'] = 'mk.bundle/1'
    request: ComputeRequest
    context: Literal['self-contained; no kernel object references or inherited assumptions'] = 'self-contained; no kernel object references or inherited assumptions'
    classification: Literal['local_only', 'explicit_export'] = 'local_only'


class RuntimeProfile(Contract):
    kind: Literal['native'] = 'native'
    digest: Digest
    python: str
    platform: str
    kernel_version: str
    dependencies: tuple[str, ...]


class SlurmAllocation(Contract):
    partition: Identifier
    allocation_account: Identifier
    qos: Identifier | None = None
    constraint: Identifier | None = None
    memory_mib: int = Field(ge=256, le=65536)
    walltime_seconds: int = Field(ge=120, le=900)
    cpus: Literal[1] = 1
    allocation_units: Literal['UNKNOWN'] = 'UNKNOWN'

    @field_validator('cpus', mode='before')
    @classmethod
    def strict_cpu_count(cls, value):
        if type(value) is not int:
            raise ValueError('CPU count requires a strict integer')
        return value


class RemoteBinding(Contract):
    target_profile_digest: Digest
    verifier_runtime: RuntimeProfile
    adapter: Literal['ssh', 'slurm']
    destination: str = Field(min_length=1, max_length=260)
    ssh_account: Identifier
    allocation: SlurmAllocation | None = None


class ExecutionSpec(Contract):
    protocol_version: Literal['1.0'] = '1.0'
    bundle: InputBundle
    bundle_digest: Digest
    target: Identifier = 'local-cpu'
    runtime: RuntimeProfile
    policy_digest: Digest
    remote: RemoteBinding | None = None

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.remote is None:
            data.pop('remote', None)  # Preserve canonical identities of existing 1.0 local records.
        return data


class Money(Contract):
    currency: Literal['EUR'] = 'EUR'
    amount: Annotated[str, Field(pattern=r'^(0|[1-9][0-9]*)\.[0-9]{2}$', max_length=32)] = '0.00'


class ComputePlan(Contract):
    plan_id: Identifier
    workspace_id: Identifier
    created_ms: int
    expires_ms: int
    spec: ExecutionSpec
    execution_digest: Digest
    provider_cost: Money | None = Field(default_factory=Money)
    warnings: tuple[str, ...]
    rejected_targets: tuple[str, ...] = ('Remote/GPU adapters are not installed or authorized in this tier.',)
    authorization_required: Literal[True] = True

    @property
    def digest(self):
        return digest(self)


class AuthorizationGrant(Contract):
    grant_id: Identifier
    issuer: Literal['local-host'] = 'local-host'
    subject: Identifier
    workspace_id: Identifier
    plan_digest: Digest
    expires_ms: int
    max_attempts: Literal[1] = 1
    max_amount: Money = Field(default_factory=Money)
    export_allowed: bool = False
    provisioning_allowed: Literal[False] = False
    retries_allowed: Literal[False] = False


class AttemptRecord(Contract):
    attempt_id: Identifier
    execution_digest: Digest
    bundle_digest: Digest
    workspace_id: Identifier
    spec: ExecutionSpec
    start_deadline_ms: int | None = Field(default=None, ge=0)

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.start_deadline_ms is None:
            data.pop('start_deadline_ms', None)
        return data


class ResourceLease(Contract):
    lease_id: Identifier
    attempt_id: Identifier
    ownership: Literal['job-process'] = 'job-process'
    generation: Identifier
    expiry_ms: int
    cleanup: str


class RemoteResultEnvelope(Contract):
    """Candidate transport DTO. It deliberately imports no evidence/result models."""
    protocol_version: Literal['1.0'] = '1.0'
    attempt_id: Identifier
    workspace_id: Identifier
    bundle_digest: Digest
    execution_digest: Digest
    operation: Literal['cuboid_sweep', 'signal_convolve']
    output_schema: Literal['mk.cuboid-pairs/1', 'mk.real-convolution/1']
    output: tuple[tuple[IntegerText, IntegerText], ...] | tuple[DecimalText, ...]
    worker_claims: dict = Field(default_factory=dict)


class VerificationReport(Contract):
    report_id: Identifier
    verifier: str
    verifier_version: Literal['1'] = '1'
    candidate_digest: Digest
    bundle_digest: Digest
    execution_digest: Digest
    claim: str
    outcome: Literal['PASSED', 'FAILED', 'INCONCLUSIVE', 'UNSUPPORTED']
    trust: Literal['exact', 'numeric', 'unknown']
    detail: str
    maximum_error: str | None = None
    tolerance: str | None = None


class ComputeResultReceipt(Contract):
    job_id: Identifier
    execution: str
    verification: str
    artifacts: str
    resources: str
    cost: Literal['NO_METERED_ALLOCATION', 'ALLOCATION_USAGE_UNKNOWN', 'EXISTING_HOST_COST_UNMEASURED'] = 'NO_METERED_ALLOCATION'
    candidate_ref: Digest | None = None
    accepted_result_ref: Digest | None = None
    transport: Literal['AVAILABLE', 'UNAVAILABLE'] = 'AVAILABLE'


class JobHandle(Contract):
    job_id: Identifier
    attempt_id: Identifier
    client_request_id: Identifier


class LocalPolicy(Contract):
    enabled: bool = True
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    plan_lifetime_ms: int = Field(default=300000, ge=1000, le=900000)
    max_retained_jobs: int = Field(default=1000, ge=1, le=10000)


class JobRecord(Contract):
    job_id: Identifier
    attempt_id: Identifier
    client_request_id: Identifier
    workspace_id: Identifier
    plan_id: Identifier
    authorization_ref: Identifier
    fingerprint: Digest
    revision: int = Field(ge=0)
    observation_revision: int = Field(ge=-1)
    execution: Literal['PREPARED', 'SUBMITTING', 'SUBMISSION_UNKNOWN', 'QUEUED', 'RUNNING', 'RECEIVED', 'FAILED', 'TIMED_OUT', 'CANCELLED', 'LOST', 'OUT_OF_MEMORY', 'PREEMPTED', 'NODE_FAILED', 'EXPIRED']
    verification: Literal['NOT_REQUESTED', 'PENDING', 'RUNNING', 'PASSED', 'FAILED', 'INCONCLUSIVE', 'UNSUPPORTED']
    artifacts: Literal['NONE', 'UNAVAILABLE', 'REJECTED', 'QUARANTINED']
    resources: Literal['ALLOCATION_INTENT', 'ACTIVE', 'NOT_OWNED', 'RELEASE_CONFIRMED', 'CLEANUP_UNKNOWN']
    candidate_ref: Digest | None = None
    accepted_result_ref: Digest | None = None
    transport: Literal['AVAILABLE', 'UNAVAILABLE'] = 'AVAILABLE'
    cancel_requested: bool
    message: str = Field(max_length=2000)


class OperationDescriptor(Contract):
    operation: Literal['cuboid_sweep', 'signal_convolve']
    version: Literal['1'] = '1'
    engine: Literal['python', 'scipy']
    output_schema: Literal['mk.cuboid-pairs/1', 'mk.real-convolution/1']
    arithmetic: str
    claims: tuple[str, ...]
    verifier: str
    parameter_schema_digest: Digest
    side_effects: Literal['none'] = 'none'
