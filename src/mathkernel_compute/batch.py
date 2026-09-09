"""Bounded independent tasks, with stable shard identity and no implicit reducer."""
from pydantic import Field, model_validator
from .models import Contract, Identifier, Digest, ComputeRequest, ComputePlan, JobHandle
from .protocol import digest


class BatchShard(Contract):
    shard_id: Identifier
    request: ComputeRequest


class BatchRequest(Contract):
    shards: tuple[BatchShard, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def independent(self):
        if len({s.shard_id for s in self.shards}) != len(self.shards):
            raise ValueError('DUPLICATE_LOGICAL_SHARD')
        if len({s.request.target for s in self.shards}) != 1:
            raise ValueError('BATCH_REQUIRES_ONE_PINNED_TARGET')
        return self


class PlannedShard(Contract):
    shard_id: Identifier
    plan: ComputePlan


class BatchPlan(Contract):
    batch_plan_id: Identifier
    workspace_id: Identifier
    shards: tuple[PlannedShard, ...] = Field(min_length=1, max_length=16)
    expires_ms: int
    semantics: str = 'Independent registered tasks; no graph, shared memory, statistical independence claim or mathematical reducer.'

    @property
    def digest(self):
        return digest(self)


class BatchGrant(Contract):
    grant_id: Identifier
    workspace_id: Identifier
    batch_plan_digest: Digest
    subject: Identifier
    expires_ms: int
    export_allowed: bool
    managed_exposure_acknowledged: bool
    persistent_storage_allowed: bool


class ShardHandle(Contract):
    shard_id: Identifier
    job: JobHandle


class BatchHandle(Contract):
    batch_id: Identifier
    client_request_id: Identifier
    shards: tuple[ShardHandle, ...] = Field(min_length=1, max_length=16)
