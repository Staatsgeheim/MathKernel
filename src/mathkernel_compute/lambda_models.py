"""Owned-VM authority is separate from mathematical requests and execution grants."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from .models import Contract, Identifier, Digest, ManagedQuote, BudgetLimit, RuntimeProfile
from .protocol import digest

ProviderID = Annotated[str, Field(pattern=r'^(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$')]


class LambdaProfile(Contract):
    target_id: Identifier
    adapter: Literal['lambda'] = 'lambda'
    account_scope: Identifier
    budget_id: Identifier
    region_name: Identifier
    instance_type_name: Identifier
    image_id: ProviderID
    ssh_key_name: Identifier
    firewall_ruleset_id: ProviderID
    # No cloud-init, arbitrary command, downloaded code, secret or filesystem mounts.
    runtime: RuntimeProfile
    quote: ManagedQuote
    lease_ms: int = Field(default=900000, ge=120000, le=3600000)
    credential_env: str = Field(default='MK_LAMBDA_API_KEY', pattern=r'^MK_[A-Z0-9_]{1,64}_API_KEY$')
    cleanup_policy: Literal['strict', 'managed-exposure'] = 'strict'
    qualification: Literal['experimental-not-live-qualified'] = 'experimental-not-live-qualified'

    @model_validator(mode='after')
    def target(self):
        if self.target_id in {'auto', 'local-cpu'}:
            raise ValueError('Reserved target alias')
        if self.runtime.accelerator is not None:
            raise ValueError('Lambda delegates to the current CPU-only native SSH profile')
        return self

    @property
    def profile_digest(self):
        return digest(self)


class VMPlan(Contract):
    schema_version: Literal['mk.vm-plan/1'] = 'mk.vm-plan/1'
    lease_id: Identifier
    workspace_id: Identifier
    profile: LambdaProfile
    budget: BudgetLimit
    created_ms: int = Field(ge=0)
    launch_before_ms: int = Field(ge=0)
    cleanup_after_ms: int = Field(ge=0)
    provisioning_allowed: Literal[False] = False  # A plan is never authority.
    native_expiry_supported: Literal[False] = False
    hard_spend_cap_supported: Literal[False] = False
    input_export: Literal['separate-SSH-plan-and-grant-required'] = 'separate-SSH-plan-and-grant-required'

    @model_validator(mode='after')
    def scope(self):
        if (self.budget.budget_id != self.profile.budget_id or self.budget.account_scope != self.profile.account_scope
                or not self.created_ms < self.launch_before_ms <= self.created_ms + 900000
                or self.launch_before_ms > self.profile.quote.valid_until_ms
                or self.cleanup_after_ms != self.launch_before_ms + self.profile.lease_ms):
            raise ValueError('VM_PLAN_SCOPE')
        return self

    @property
    def digest(self):
        return digest(self)

    @property
    def tags(self):
        return {'mk-workspace': self.workspace_id, 'mk-lease': self.lease_id, 'mk-plan': self.digest}


class VMGrant(Contract):
    grant_id: Identifier
    lease_id: Identifier
    plan_digest: Digest
    workspace_id: Identifier
    expires_ms: int = Field(ge=0)
    provisioning_allowed: Literal[True]
    managed_exposure_acknowledged: Literal[True]
    broad_key_privileges_acknowledged: Literal[True]
    network_egress_acknowledged: Literal[True]
    possible_output_loss_at_deadline_acknowledged: Literal[True]
    issuer: Literal['local-host'] = 'local-host'
    max_instances: Literal[1] = 1
    retries_allowed: Literal[False] = False


class WatchdogTicket(Contract):
    schema_version: Literal['mk.vm-watchdog/1'] = 'mk.vm-watchdog/1'
    plan: VMPlan
    # This data file does not authorize termination until approved on the watchdog host.
    authority: Literal['requires-independent-host-approval'] = 'requires-independent-host-approval'

    @property
    def digest(self):
        return digest(self)
