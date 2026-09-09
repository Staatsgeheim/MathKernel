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
    engine: Literal['python', 'cuda'] = 'python'

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
    gpu_count: Literal[0, 1] = 0

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
        wants_gpu = isinstance(self.parameters, CuboidParameters) and self.parameters.engine == 'cuda'
        if self.resources.gpu_count != int(wants_gpu):
            raise ValueError('CUDA engine requires exactly one GPU; CPU engines require zero GPUs')
        return self


class TargetCapabilities(Contract):
    process_deadline: Literal['supported', 'unsupported']
    process_tree_cleanup: Literal['supported', 'unsupported']
    output_retention: Literal['controller-spool', 'remote-spool', 'durable-provider-store'] = 'controller-spool'
    network_isolation: Literal['unsupported', 'provider-configured'] = 'unsupported'
    hard_memory_limit: Literal['unsupported', 'provider-configured'] = 'unsupported'
    hard_cost_cap: Literal['unsupported'] = 'unsupported'
    source: Literal['local-runtime', 'operator-profile'] = 'local-runtime'


class ComputeTarget(Contract):
    target_id: Identifier = 'local-cpu'
    adapter: Literal['local', 'ssh', 'slurm', 'modal', 'runpod'] = 'local'
    ownership: Literal['existing'] = 'existing'
    available: bool
    capabilities: TargetCapabilities
    reason: str


class InputBundle(Contract):
    bundle_schema: Literal['mk.bundle/1'] = 'mk.bundle/1'
    request: ComputeRequest
    context: Literal['self-contained; no kernel object references or inherited assumptions'] = 'self-contained; no kernel object references or inherited assumptions'
    classification: Literal['local_only', 'explicit_export'] = 'local_only'


class AcceleratorProfile(Contract):
    cupy_version: str = Field(max_length=64)
    cuda_runtime: int = Field(ge=10000, le=99999)
    cuda_driver: int = Field(ge=10000, le=99999)


class RuntimeProfile(Contract):
    kind: Literal['native'] = 'native'
    digest: Digest
    python: str
    platform: str
    kernel_version: str
    dependencies: tuple[str, ...]
    accelerator: AcceleratorProfile | None = None

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.accelerator is None:
            data.pop('accelerator', None)
        return data


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


class Money(Contract):
    currency: Literal['EUR', 'USD'] = 'EUR'
    amount: Annotated[str, Field(pattern=r'^(0|[1-9][0-9]*)\.[0-9]{2}$', max_length=32)] = '0.00'


class ManagedQuote(Contract):
    quote_id: Identifier
    reservation: Money
    valid_until_ms: int = Field(ge=0)
    source: str = Field(min_length=1, max_length=512)
    scope: str = Field(min_length=1, max_length=1000)
    hard_spend_cap_supported: Literal[False] = False

    @model_validator(mode='after')
    def native_currency(self):
        if self.reservation.currency != 'USD' or not Decimal('0.01') <= Decimal(self.reservation.amount) <= Decimal('1000000'):
            raise ValueError('Managed reservations require a positive bounded USD amount')
        return self


class BudgetLimit(Contract):
    budget_id: Identifier
    account_scope: Identifier
    limit: Money

    @model_validator(mode='after')
    def native_currency(self):
        if self.limit.currency != 'USD' or not Decimal('0.01') <= Decimal(self.limit.amount) <= Decimal('1000000'):
            raise ValueError('Managed budget limits require a positive bounded USD amount')
        return self


class ManagedResources(Contract):
    cpu_millicores: int = Field(default=1000, ge=1000, le=16000)
    memory_mib: int = Field(default=2048, ge=512, le=65536)
    gpu: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    lifetime_ms: int = Field(default=180000, ge=120000, le=900000)
    ttl_ms: int = Field(default=600000, ge=120000, le=3600000)
    network: Literal['blocked', 'broker-only'] = 'blocked'

    @model_validator(mode='after')
    def finite_lifetime(self):
        if self.ttl_ms < self.lifetime_ms:
            raise ValueError('Total TTL must cover maximum active lifetime')
        return self


class ModalOptions(Contract):
    app_name: Identifier
    environment: Identifier
    region: str
    sdk_version: Literal['1.5.5'] = '1.5.5'
    volume_version: Literal[2] = 2


class RunpodOptions(Contract):
    deployment_digest: Digest
    workers_min: int = Field(ge=0, le=16)
    workers_max: int = Field(ge=1, le=16)
    idle_timeout_seconds: int = Field(ge=1, le=600)


class ManagedBinding(Contract):
    adapter: Literal['modal', 'runpod']
    target_profile_digest: Digest
    verifier_runtime: RuntimeProfile
    account_scope: Identifier
    destination: str = Field(min_length=1, max_length=256)
    image_identity: str = Field(min_length=1, max_length=256)
    storage_scope: str = Field(min_length=1, max_length=512)
    source_image: str = Field(min_length=1, max_length=256)
    provider_options: ModalOptions | RunpodOptions
    resources: ManagedResources
    budget: BudgetLimit
    quote: ManagedQuote
    qualification: Literal['experimental-not-live-qualified'] = 'experimental-not-live-qualified'


class PaidApproval(Contract):
    budget_id: Identifier
    reservation: Money
    managed_exposure_acknowledged: bool
    persistent_storage_allowed: bool

    @model_validator(mode='after')
    def explicit_scope(self):
        if not self.managed_exposure_acknowledged or not self.persistent_storage_allowed:
            raise ValueError('Paid execution requires explicit exposure and storage approval')
        return self


class ExecutionSpec(Contract):
    protocol_version: Literal['1.0'] = '1.0'
    bundle: InputBundle
    bundle_digest: Digest
    target: Identifier = 'local-cpu'
    runtime: RuntimeProfile
    policy_digest: Digest
    remote: RemoteBinding | None = None
    managed: ManagedBinding | None = None

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.managed is None:
            data.pop('managed', None)
        if self.remote is None:
            data.pop('remote', None)  # Preserve canonical identities of existing 1.0 local records.
        return data

    @model_validator(mode='after')
    def single_endpoint(self):
        if self.remote is not None and self.managed is not None:
            raise ValueError('Execution has exactly one provider binding')
        return self

    @property
    def endpoint(self):
        return self.managed or self.remote


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
    paid: PaidApproval | None = None

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.paid is None:
            data.pop('paid', None)
        return data


class BatchBinding(Contract):
    batch_id: Identifier
    shard_id: Identifier
    batch_plan_digest: Digest


class AttemptRecord(Contract):
    attempt_id: Identifier
    execution_digest: Digest
    bundle_digest: Digest
    workspace_id: Identifier
    spec: ExecutionSpec
    start_deadline_ms: int | None = Field(default=None, ge=0)
    batch: BatchBinding | None = None

    @model_serializer(mode='wrap')
    def compatible_bytes(self, handler):
        data = handler(self)
        if self.batch is None:
            data.pop('batch', None)
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
    cost: Literal['NO_METERED_ALLOCATION', 'ALLOCATION_USAGE_UNKNOWN', 'EXISTING_HOST_COST_UNMEASURED', 'RESERVED', 'EXPOSURE_UNKNOWN', 'USER_RECONCILED'] = 'NO_METERED_ALLOCATION'
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
    optional_engines: tuple[Literal['cuda'], ...] = ()
    output_schema: Literal['mk.cuboid-pairs/1', 'mk.real-convolution/1']
    arithmetic: str
    claims: tuple[str, ...]
    verifier: str
    parameter_schema_digest: Digest
    side_effects: Literal['none'] = 'none'
