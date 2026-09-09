"""Reproducible local checks, NOT a verification of the Navier-Stokes construction.

Run from the repository root:
  python examples/audits/openai_navier_stokes/local_checks.py --output /path/new.json
The hypotheses/formula transcription are reviewed inputs, not proved by this run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from mathkernel import MathKernel, TrustLevel

SOURCE = 'https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf'
COMMIT = '8937a8f4cbc7abaab5e9e97d1cc7f5d2319d9538'
# Exponent arithmetic is conditional on the ansatz of the paper, pp. 3-7.
# Write leading negative rationals as (-1/2); unary minus spans a following difference.
IDENTITIES = (
    ('support_volume_exponent', '2*(1/2)+(1/2-h)', '3/2-h'),
    ('kinetic_energy_exponent', '2*((-1/2)-h)+(3/2-h)', '1/2-3*h'),
    ('L3_cubed_exponent', '3*((-1/2)-h)+(3/2-h)', '-4*h'),
    ('radial_enstrophy_scaling', '2*(((-1/2)-h)-1/2)+(3/2-h)', '(-1/2)-3*h'),
    ('swirl_reynolds_exponent', '((-1/2)-h)+1/2', '-h'),
    ('radial_advection_rate', '(-1/2)-(1/2)', '-1'),
    ('axial_advection_rate', '((-1/2)-h)-(1/2-h)', '-1'),
    ('axial_to_radial_diffusion_ratio', '-2*(1/2-h)-(-2*(1/2))', '2*h'),
    ('chart_jacobian_reduced', '(1-eta^2)+2*(1/2-h)*eta^2', '1-2*h*eta^2'),
    ('viscosity_advection_factor', 's*(s/s)', 's'),
    ('viscosity_diffusion_factor', 's^2*(s/s^2)', 's'),
    ('viscosity_pressure_gradient_factor', 's^2/s', 's'),
    ('viscosity_energy_factor', 's^2*s^3', 's^5'),
)


def rational_substitution(text: str) -> str:
    values = {'h': '1/200', 'eta': '1/2', 's': '2'}
    return re.sub(r'\b(h|eta|s)\b', lambda m: '(' + values[m.group(0)] + ')', text)


def run_checks() -> dict:
    kernel = MathKernel()
    ctx = kernel.create_context({'h': 'real', 'eta': 'real', 's': 'real'},
                                ['h > 0', 'h < 1/100', 's > 0', 'eta > -1', 'eta < 1'])
    checks, mutations = [], []
    for name, lhs, rhs in IDENTITIES:
        checked = kernel.prove_equivalence(lhs, rhs, ctx.context_id, formal=False)
        checks.append({'name': name, 'lhs': lhs, 'rhs': rhs,
                       'passed': checked.status == 'verified',
                       'result': checked.model_dump(mode='json')})
        # Negative controls are deliberately perturbed, closed rational claims.
        # They must be rejected; they are NOT alleged errors in the source paper.
        bad_lhs, bad_rhs = rational_substitution(lhs), '(' + rational_substitution(rhs) + ')+1'
        rejected = kernel.prove_equivalence(bad_lhs, bad_rhs, formal=False)
        mutations.append({'name': name + '_deliberate_plus_one', 'lhs': bad_lhs, 'rhs': bad_rhs,
                          'rejected': rejected.status == 'refuted' and rejected.trust == TrustLevel.EXACT,
                          'result': rejected.model_dump(mode='json')})
    bounds = []
    for name, expression, lo, hi in (
        ('energy_exponent_positive', '1/2-3*h', '0', '1/100'),
        ('enstrophy_time_integrability_margin', '((-1/2)-3*h)+1', '0', '1/100'),
        # eta^2 <= 1 implies J/q^D >= 1-2h. This implication is a reviewed
        # elementary bound; this interval call certifies its positive margin.
        ('jacobian_lower_margin', '1-2*h', '0', '1/100'),
    ):
        eid = kernel.parse(expression).data['expr_id']
        checked = kernel.certified_enclose(eid, 'h', lo, hi)
        bounds.append({'name': name, 'expression': expression, 'interval': [lo, hi],
                       'passed': checked.ok and checked.data.get('strictly_positive') is True,
                       'result': checked.model_dump(mode='json')})
    return {'schema_version': 'mathkernel.navier-local-audit/v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_url': SOURCE, 'formal_source_commit': COMMIT,
            'scope': 'finite algebraic/interval checks of manually transcribed scaling formulas',
            'theorem_verification': 'not_run', 'disproof_found': False,
            'hypotheses': ['0 < h < 1/100', 'q > 0', '|eta| < 1', 's = sqrt(nu) > 0',
                           'the ansatz scaling described in the paper actually holds'],
            'identities': checks, 'certified_margins': bounds, 'negative_controls': mutations,
            'summary': {'identities_passed': sum(c['passed'] for c in checks), 'identities_total': len(checks),
                        'margins_passed': sum(c['passed'] for c in bounds), 'margins_total': len(bounds),
                        'mutations_rejected': sum(c['rejected'] for c in mutations), 'mutations_total': len(mutations)},
            'limitations': ['No construction of the velocity/pressure/forcing was reproduced.',
                            'No all-order residual or convergence estimate was checked.',
                            'Formal proof compilation, axiom export and nanoda were not run.',
                            'A correct scaling identity does not establish existence of a PDE solution.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run_checks()
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result['summary'], indent=2))
    return 0 if all(c['passed'] for c in result['identities'] + result['certified_margins']) and all(c['rejected'] for c in result['negative_controls']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
