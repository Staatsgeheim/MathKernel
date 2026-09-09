"""Bounded inert replay inspection. Importing a manifest creates no authority or job."""
from typing import Literal
from pydantic import Field, model_validator
from .models import Contract, ComputePlan, AttemptRecord, ComputeResultReceipt, Digest
from .protocol import canonical, decode, digest, parse


class Replay(Contract):
    schema_version: Literal['mk.compute-replay/1', 'mk.compute-replay/2'] = Field(alias='schema')
    plan: ComputePlan
    attempt: AttemptRecord
    status: ComputeResultReceipt
    external_artifacts: tuple[Digest, ...] = ()

    @model_validator(mode='after')
    def binding(self):
        p, a = self.plan, self.attempt
        if (p.execution_digest != digest(p.spec) or p.spec.bundle_digest != digest(p.spec.bundle)
                or a.execution_digest != p.execution_digest or a.bundle_digest != p.spec.bundle_digest
                or a.spec != p.spec or a.workspace_id != p.workspace_id):
            raise ValueError('REPLAY_BINDING_MISMATCH')
        if self.schema_version == 'mk.compute-replay/2':
            expected = tuple(v for v in (self.status.candidate_ref, self.status.accepted_result_ref) if v)
            if self.external_artifacts != expected:
                raise ValueError('REPLAY_ARTIFACT_DECLARATION_MISMATCH')
        return self


def inspect_replay(raw):
    value = parse(Replay, raw)
    required = tuple(v for v in (value.status.candidate_ref, value.status.accepted_result_ref) if v)
    return {'schema': value.schema_version, 'manifest_digest': digest(decode(raw)),
        'operation': value.attempt.spec.bundle.request.operation, 'target': value.attempt.spec.target,
        'binding': 'consistent-data-only', 'mathematical_trust': 'not-established-by-replay',
        'execution_authority': 'none; a fresh plan and host grant are required',
        'external_artifacts': required, 'artifacts_included': False,
        'runtime': value.attempt.spec.runtime.model_dump(mode='json')}


def export_replay(plan, attempt, status):
    raw = canonical({'schema': 'mk.compute-replay/2', 'plan': plan.model_dump(mode='json'),
        'attempt': attempt.model_dump(mode='json'), 'status': status.model_dump(mode='json'),
        'external_artifacts': [v for v in (status.candidate_ref, status.accepted_result_ref) if v]})
    inspect_replay(raw)
    return raw
