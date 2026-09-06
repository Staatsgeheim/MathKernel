# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Bounded Phase F.3 rank tests and replayable resampling inference."""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import mpmath as mp
import numpy as np
import sympy as sp

from mathkernel_artifacts import (
    ComputationEvidence, EmpiricalEvidence, EvidenceBundle, ModelEvidence,
    NumericalEvidence,
)

from .engineering import EngineeringResult, cap_trust
from .statistical_inference import (
    NonparametricTestResult, ResamplingResult, StatisticalSample, _compare,
    _quantile, _sorted,
)


ALTERNATIVES = {"two_sided", "less", "greater"}
METHODS = {"auto", "exact", "asymptotic"}


def _column(sample: StatisticalSample, name: str) -> tuple[sp.Expr, ...]:
    if name not in sample.variables:
        raise ValueError(f"{name!r} must name a stored sample variable")
    index = sample.variables.index(name)
    return tuple(row[index] for row in sample.observations)


def _equal(left: sp.Expr, right: sp.Expr) -> bool:
    return _compare(left, right) == 0


def _unique(values: Iterable[sp.Expr]) -> tuple[sp.Expr, ...]:
    ordered = _sorted(tuple(values)); output = []
    for value in ordered:
        if not output or not _equal(output[-1], value):
            output.append(value)
    return tuple(output)


def _ranks(values: tuple[sp.Expr, ...]):
    indexed = sorted(enumerate(values), key=lambda item: _RankKey(item[1]))
    ranks = [sp.S.Zero] * len(values); ties = []
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and _equal(indexed[start][1], indexed[end][1]):
            end += 1
        rank = sp.Rational(start + 1 + end, 2)
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        if end - start > 1:
            ties.append(end - start)
        start = end
    return tuple(ranks), tuple(ties)


class _RankKey:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return _compare(self.value, other.value) < 0


def _mp_expr(value) -> sp.Float:
    return sp.Float(str(value), 50)


def _input_trust(sample: StatisticalSample, *parameters) -> str:
    return cap_trust(sample.input_trust,
                     "numeric" if any(isinstance(item, sp.Basic) and item.has(sp.Float)
                                      for item in parameters) else "exact")


def _normal_cdf(z):
    return (1 + mp.erf(z / mp.sqrt(2))) / 2


def _normal_p(z, alternative: str):
    if alternative == "two_sided":
        return min(mp.mpf(1), 2 * (1 - _normal_cdf(abs(z))))
    if alternative == "greater":
        return 1 - _normal_cdf(z)
    return _normal_cdf(z)


def _extreme(statistic, observed, center, alternative: str) -> bool:
    if alternative == "greater":
        difference = sp.simplify(statistic - observed)
    elif alternative == "less":
        difference = sp.simplify(observed - statistic)
    else:
        difference = sp.simplify(abs(statistic - center) -
                                 abs(observed - center))
    if difference.is_nonnegative is not None:
        return bool(difference.is_nonnegative)
    if not difference.has(sp.Float):
        raise ValueError("exact extremeness comparison is undecidable")
    left = float(statistic); right = float(observed); middle = float(center)
    if alternative == "greater":
        return left >= right - 1e-14
    if alternative == "less":
        return left <= right + 1e-14
    return abs(left - middle) >= abs(right - middle) - 1e-14


def _bounded_comb(n: int, k: int, maximum: int) -> int:
    k = min(k, n - k); value = 1
    for index in range(1, k + 1):
        value = value * (n - k + index) // index
        if value > maximum:
            return maximum + 1
    return value


def _bounded_factorial(n: int, maximum: int) -> int:
    value = 1
    for item in range(2, n + 1):
        value *= item
        if value > maximum:
            return maximum + 1
    return value


def _bounded_multinomial(sizes: tuple[int, ...], maximum: int) -> int:
    remaining = sum(sizes); value = 1
    for size in sizes[:-1]:
        factor = _bounded_comb(remaining, size, maximum)
        value *= factor
        if value > maximum:
            return maximum + 1
        remaining -= size
    return value


def _assumptions(test: str) -> list[str]:
    return {
        "mann_whitney": ["group labels are exchangeable under the null",
                         "observations are independent across groups"],
        "wilcoxon": ["paired differences are symmetric under the null",
                     "nonzero pairs are exchangeable under sign reversal"],
        "kruskal_wallis": ["group labels are exchangeable under the null",
                           "observations are independent across groups"],
        "ks_2samp": ["samples are independent under the null",
                     "pooled labels are exchangeable under the null"],
        "spearman": ["paired observations are exchangeable under independence"],
        "kendall": ["paired observations are exchangeable under independence"],
        "permutation_test": ["group labels are exchangeable under the null"],
        "bootstrap": ["the empirical distribution is a suitable resampling proxy"],
    }[test]


def _bundle(sample: StatisticalSample, test: str, method: str, trust: str, *,
            numerical=None, simulation=None) -> EvidenceBundle:
    empirical = [EmpiricalEvidence(
        sample_size=len(sample.observations), sampling_method=sample.sampling_method,
        role="diagnostic", trust="empirical",
        metadata={"scope": "stored observations only"})]
    if simulation is not None:
        empirical.append(EmpiricalEvidence(
            sample_size=simulation["resamples"], sampling_method="PCG64",
            replication={"seed": simulation["seed"],
                         "algorithm": "PCG64",
                         "resamples": simulation["resamples"]},
            uncertainty=simulation.get("monte_carlo_standard_error"),
            role="required", trust="empirical",
            metadata={"scope": "resampling approximation"}))
    bundle = EvidenceBundle(
        computation=[ComputationEvidence(
            engine="statistical_inference", method=method, arithmetic=trust,
            deterministic=simulation is None, trust=trust,
            metadata={"inference_method": method})],
        empirical=empirical,
        model=[ModelEvidence(
            assumptions=_assumptions(test),
            diagnostics=["null/resampling assumptions are not established",
                         "population generalization is not established",
                         "causal interpretation is not established"],
            role="diagnostic", trust="unknown",
            metadata={"test": test, "population_claim": "not_established",
                      "causal_claim": "not_established"})],
        justified_trust=trust)
    if numerical is not None:
        bundle.numerical.append(NumericalEvidence(
            precision=166, residual=None, trust=trust, role="required",
            metadata={"approximation": numerical, "error_bound": False}))
    return bundle


def _result(sample, operation, output, checks, *, method, trust,
            numerical=None, simulation=None):
    checks = {name: (None if accepted is None else bool(accepted))
              for name, accepted in checks.items()}
    bundle = _bundle(sample, operation, method, trust,
                     numerical=numerical, simulation=simulation)
    return EngineeringResult(
        operation=operation, status="verified", trust=trust,
        value={"test": operation,
               "statistic": output.statistic if isinstance(output, NonparametricTestResult)
               else output.observed_statistic,
               "p_value": output.p_value,
               "method": output.method,
               "enumeration_or_resample_count": output.enumeration_count
               if isinstance(output, NonparametricTestResult) else output.resamples},
        details={"identity_checks": checks,
                 "inference_scope": output.inference_scope,
                 "null_or_resampling_assumptions": "asserted_not_verified",
                 "population_generalization": "not_established",
                 "causal_effect": "not_established"},
        verification=checks, claim_evidence={operation: bundle}), output


def _select_method(requested: str, states: int, maximum: int) -> str:
    if requested not in METHODS:
        raise ValueError("method must be auto, exact, or asymptotic")
    if requested == "exact" and states > maximum:
        raise ValueError("exact enumeration exceeds max_exact_resampling_states")
    return "exact" if requested == "exact" or (
        requested == "auto" and states <= maximum) else "asymptotic"


def mann_whitney(sample: StatisticalSample, value: str, group: str,
                 group_a: sp.Expr, group_b: sp.Expr, *, alternative="two_sided",
                 method="auto", max_exact_states=100_000, max_work=20_000_000):
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be two_sided, less, or greater")
    if _equal(group_a, group_b):
        raise ValueError("group_a and group_b must differ")
    values = _column(sample, value); labels = _column(sample, group)
    left = tuple(item for item, label in zip(values, labels) if _equal(label, group_a))
    right = tuple(item for item, label in zip(values, labels) if _equal(label, group_b))
    if any(not (_equal(label, group_a) or _equal(label, group_b)) for label in labels):
        raise ValueError("group column contains values outside group_a/group_b")
    if not left or not right:
        raise ValueError("both groups must contain observations")
    pooled = left + right; ranks, ties = _ranks(pooled)
    n1, n2 = len(left), len(right)
    observed = sp.simplify(sum(ranks[:n1], sp.S.Zero) -
                           sp.Rational(n1 * (n1 + 1), 2))
    center = sp.Rational(n1 * n2, 2)
    states = _bounded_comb(n1 + n2, n1, max_exact_states)
    selected = _select_method(method, states, max_exact_states)
    if selected == "exact" and states * len(pooled) > max_work:
        if method == "auto":
            selected = "asymptotic"
        else:
            raise ValueError("exact Mann-Whitney enumeration exceeds max_resampling_work")
    if selected == "exact":
        extreme = 0
        for indices in itertools.combinations(range(n1 + n2), n1):
            statistic = sum((ranks[index] for index in indices), sp.S.Zero) - sp.Rational(n1 * (n1 + 1), 2)
            extreme += _extreme(statistic, observed, center, alternative)
        p_value = sp.Rational(extreme, states); method_name = "exact_permutation"
        trust = _input_trust(sample, group_a, group_b); numerical = None
    else:
        tie_sum = sum(size ** 3 - size for size in ties)
        variance = sp.Rational(n1 * n2, 12) * (
            n1 + n2 + 1 - sp.Rational(tie_sum, (n1 + n2) * (n1 + n2 - 1)))
        if variance <= 0:
            raise ValueError("Mann-Whitney asymptotic variance is zero")
        direction = float(observed - center)
        correction = .5 * (1 if direction > 0 else -1 if direction < 0 else 0)
        z = (direction - correction) / math.sqrt(float(variance))
        p_value = _mp_expr(_normal_p(z, alternative)); method_name = "asymptotic_normal"
        trust = cap_trust(_input_trust(sample, group_a, group_b), "numeric_high_precision")
        numerical = "tie-corrected normal with continuity correction"
    output = NonparametricTestResult(
        test="mann_whitney", statistic=observed, p_value=p_value,
        alternative=alternative, method=method_name, variables=(value, group),
        sample_sizes=(n1, n2), enumeration_count=states if selected == "exact" else None,
        tie_correction=sp.Integer(sum(size ** 3 - size for size in ties)),
        configuration={"value": value, "group": group, "group_a": group_a,
                       "group_b": group_b, "alternative": alternative,
                       "method": selected}, input_trust=trust)
    checks = {"rank_sum_reconciled": sum(ranks, sp.S.Zero) ==
              sp.Rational((n1 + n2) * (n1 + n2 + 1), 2),
              "p_value_bounded": 0 <= p_value <= 1,
              "groups_partition_sample": n1 + n2 == len(values)}
    return _result(sample, "mann_whitney", output, checks, method=method_name,
                   trust=trust, numerical=numerical)


def wilcoxon(sample: StatisticalSample, left_name: str, right_name: str, *,
             alternative="two_sided", method="auto", max_exact_states=100_000,
             max_work=20_000_000):
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be two_sided, less, or greater")
    left = _column(sample, left_name); right = _column(sample, right_name)
    differences = tuple(a - b for a, b in zip(left, right) if not _equal(a, b))
    if not differences:
        raise ValueError("Wilcoxon requires at least one nonzero paired difference")
    absolute = tuple(abs(item) for item in differences)
    ranks, ties = _ranks(absolute); count = len(differences)
    observed = sp.simplify(sum((rank for rank, difference in zip(ranks, differences)
                                if difference.is_positive is True), sp.S.Zero))
    if any(item.is_positive is not True and item.is_negative is not True
           for item in differences):
        raise ValueError("paired difference signs must be decidable")
    total_rank = sp.Rational(count * (count + 1), 2); center = total_rank / 2
    states = ((1 << count) if count <= max_exact_states.bit_length()
              else max_exact_states + 1)
    selected = _select_method(method, states, max_exact_states)
    if selected == "exact" and states * count > max_work:
        if method == "auto":
            selected = "asymptotic"
        else:
            raise ValueError("exact Wilcoxon enumeration exceeds max_resampling_work")
    if selected == "exact":
        extreme = 0
        for mask in range(states):
            statistic = sum((rank for index, rank in enumerate(ranks)
                             if mask & (1 << index)), sp.S.Zero)
            extreme += _extreme(statistic, observed, center, alternative)
        p_value = sp.Rational(extreme, states); method_name = "exact_permutation"
        trust = sample.input_trust; numerical = None
    else:
        tie_term = sum(size ** 3 - size for size in ties)
        variance = (count * (count + 1) * (2 * count + 1) - tie_term / 2) / 24
        if variance <= 0:
            raise ValueError("Wilcoxon asymptotic variance is zero")
        direction = float(observed - center)
        correction = .5 * (1 if direction > 0 else -1 if direction < 0 else 0)
        z = (direction - correction) / math.sqrt(variance)
        p_value = _mp_expr(_normal_p(z, alternative)); method_name = "asymptotic_normal"
        trust = cap_trust(sample.input_trust, "numeric_high_precision")
        numerical = "tie-corrected normal with continuity correction"
    output = NonparametricTestResult(
        test="wilcoxon", statistic=observed, p_value=p_value,
        alternative=alternative, method=method_name,
        variables=(left_name, right_name), sample_sizes=(count,),
        enumeration_count=states if selected == "exact" else None,
        tie_correction=sp.Integer(sum(size ** 3 - size for size in ties)),
        configuration={"left": left_name, "right": right_name,
                       "alternative": alternative, "method": selected},
        input_trust=trust)
    checks = {"signed_ranks_bounded": 0 <= observed <= total_rank,
              "p_value_bounded": 0 <= p_value <= 1,
              "zero_differences_removed_explicitly": True}
    return _result(sample, "wilcoxon", output, checks, method=method_name,
                   trust=trust, numerical=numerical)


def _kruskal_statistic(ranks, allocations, tie_correction):
    total = len(ranks)
    raw = sp.Rational(12, total * (total + 1)) * sum(
        sp.simplify(sum((ranks[index] for index in group), sp.S.Zero) ** 2 / len(group))
        for group in allocations) - 3 * (total + 1)
    return sp.simplify(raw / tie_correction)


def _allocations(indices: tuple[int, ...], sizes: tuple[int, ...]):
    if len(sizes) == 1:
        yield (indices,); return
    for chosen in itertools.combinations(indices, sizes[0]):
        selected = set(chosen)
        remaining = tuple(index for index in indices if index not in selected)
        for tail in _allocations(remaining, sizes[1:]):
            yield (tuple(chosen), *tail)


def kruskal_wallis(sample: StatisticalSample, value: str, group: str, *,
                   groups: tuple[sp.Expr, ...] | None = None, method="auto",
                   max_exact_states=100_000, max_work=20_000_000,
                   max_groups=64):
    values = _column(sample, value); labels = _column(sample, group)
    groups = groups or _unique(labels)
    if len(groups) < 2 or len(groups) > max_groups or len(_unique(groups)) != len(groups):
        raise ValueError("Kruskal-Wallis requires at least two unique groups")
    if any(not any(_equal(label, item) for item in groups) for label in labels):
        raise ValueError("group column contains values outside groups")
    allocations = tuple(tuple(index for index, label in enumerate(labels)
                              if _equal(label, item)) for item in groups)
    if any(not allocation for allocation in allocations):
        raise ValueError("every requested group must contain observations")
    ranks, ties = _ranks(values); total = len(values)
    correction = sp.S.One - sp.Rational(sum(size ** 3 - size for size in ties),
                                         total ** 3 - total)
    if correction == 0:
        raise ValueError("Kruskal-Wallis is undefined when all values tie")
    observed = _kruskal_statistic(ranks, allocations, correction)
    sizes = tuple(len(item) for item in allocations)
    states = _bounded_multinomial(sizes, max_exact_states)
    selected = _select_method(method, states, max_exact_states)
    if selected == "exact" and states * total > max_work:
        if method == "auto":
            selected = "asymptotic"
        else:
            raise ValueError("exact Kruskal-Wallis enumeration exceeds max_resampling_work")
    if selected == "exact":
        extreme = sum(_extreme(_kruskal_statistic(ranks, allocation, correction),
                               observed, 0, "greater")
                      for allocation in _allocations(tuple(range(total)), sizes))
        p_value = sp.Rational(extreme, states); method_name = "exact_permutation"
        trust = _input_trust(sample, *groups); numerical = None
    else:
        with mp.workdps(50):
            p_value = _mp_expr(mp.gammainc((len(groups) - 1) / 2,
                                           float(observed) / 2, mp.inf,
                                           regularized=True))
        method_name = "asymptotic_chi_square"
        trust = cap_trust(_input_trust(sample, *groups), "numeric_high_precision")
        numerical = "tie-corrected chi-square approximation"
    output = NonparametricTestResult(
        test="kruskal_wallis", statistic=observed, p_value=p_value,
        method=method_name, variables=(value, group), sample_sizes=sizes,
        enumeration_count=states if selected == "exact" else None,
        tie_correction=correction,
        configuration={"value": value, "group": group, "groups": groups,
                       "method": selected}, input_trust=trust)
    checks = {"groups_partition_sample": sum(sizes) == total,
              "tie_correction_positive": correction > 0,
              "p_value_bounded": 0 <= p_value <= 1}
    return _result(sample, "kruskal_wallis", output, checks, method=method_name,
                   trust=trust, numerical=numerical)


def _ks_statistic(left, right, alternative):
    support = _unique(tuple(left) + tuple(right)); n1, n2 = len(left), len(right)
    plus = minus = sp.S.Zero
    for point in support:
        f1 = sp.Rational(sum(_compare(item, point) <= 0 for item in left), n1)
        f2 = sp.Rational(sum(_compare(item, point) <= 0 for item in right), n2)
        plus = max(plus, f1 - f2); minus = max(minus, f2 - f1)
    return plus if alternative == "greater" else minus if alternative == "less" else max(plus, minus)


def ks_2samp(sample: StatisticalSample, left_name: str, right_name: str, *,
             alternative="two_sided", method="auto", max_exact_states=100_000,
             max_work=20_000_000):
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be two_sided, less, or greater")
    left = _column(sample, left_name); right = _column(sample, right_name)
    observed = _ks_statistic(left, right, alternative)
    pooled = left + right; n1, n2 = len(left), len(right)
    states = _bounded_comb(n1 + n2, n1, max_exact_states)
    selected = _select_method(method, states, max_exact_states)
    if selected == "exact" and states * len(pooled) ** 2 > max_work:
        if method == "auto":
            selected = "asymptotic"
        else:
            raise ValueError("exact KS enumeration exceeds max_resampling_work")
    if selected == "exact":
        extreme = 0; all_indices = set(range(n1 + n2))
        for chosen in itertools.combinations(range(n1 + n2), n1):
            chosen_set = set(chosen)
            statistic = _ks_statistic(tuple(pooled[index] for index in chosen),
                                      tuple(pooled[index] for index in all_indices - chosen_set),
                                      alternative)
            extreme += _extreme(statistic, observed, 0, "greater")
        p_value = sp.Rational(extreme, states); method_name = "exact_permutation"
        trust = cap_trust(sample.input_trust, "exact"); numerical = None
    else:
        effective = n1 * n2 / (n1 + n2); scaled = math.sqrt(effective) * float(observed)
        if alternative == "two_sided":
            with mp.workdps(50):
                probability = 2 * sum((-1) ** (index - 1) * mp.exp(-2 * index ** 2 * scaled ** 2)
                                      for index in range(1, 101))
                p_value = _mp_expr(max(mp.mpf(0), min(mp.mpf(1), probability)))
        else:
            p_value = _mp_expr(mp.exp(-2 * scaled ** 2))
        method_name = "asymptotic_kolmogorov"
        trust = cap_trust(sample.input_trust, "numeric_high_precision")
        numerical = "two-sample Kolmogorov limiting distribution"
    output = NonparametricTestResult(
        test="ks_2samp", statistic=observed, p_value=p_value,
        alternative=alternative, method=method_name,
        variables=(left_name, right_name), sample_sizes=(n1, n2),
        enumeration_count=states if selected == "exact" else None,
        configuration={"left": left_name, "right": right_name,
                       "alternative": alternative, "method": selected},
        input_trust=trust)
    checks = {"ecdf_distance_bounded": 0 <= observed <= 1,
              "p_value_bounded": 0 <= p_value <= 1,
              "pooled_size_reconciled": len(pooled) == n1 + n2}
    return _result(sample, "ks_2samp", output, checks, method=method_name,
                   trust=trust, numerical=numerical)


def _spearman_statistic(left, right):
    rank_left, _ = _ranks(tuple(left)); rank_right, _ = _ranks(tuple(right))
    mean = sp.Rational(len(left) + 1, 2)
    centered_left = tuple(item - mean for item in rank_left)
    centered_right = tuple(item - mean for item in rank_right)
    denominator = sp.sqrt(sum(item ** 2 for item in centered_left) *
                          sum(item ** 2 for item in centered_right))
    if denominator == 0:
        raise ValueError("Spearman correlation is undefined for a constant variable")
    return sp.simplify(sum(a * b for a, b in zip(centered_left, centered_right)) /
                       denominator)


def _kendall_statistic(left, right):
    concordant = discordant = ties_left = ties_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = _compare(left[i], left[j]); dy = _compare(right[i], right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_left += 1
            elif dy == 0:
                ties_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = sp.sqrt((concordant + discordant + ties_left) *
                          (concordant + discordant + ties_right))
    if denominator == 0:
        raise ValueError("Kendall tau is undefined for a constant variable")
    return sp.simplify(sp.Rational(concordant - discordant, 1) / denominator), (
        concordant - discordant)


def _association(sample, left_name, right_name, *, test, alternative,
                 method, max_exact_states, max_work):
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be two_sided, less, or greater")
    left = _column(sample, left_name); right = _column(sample, right_name)
    count = len(left)
    if count < 3:
        raise ValueError(f"{test} requires at least three pairs")
    statistic_fn = _spearman_statistic if test == "spearman" else lambda a, b: _kendall_statistic(a, b)[0]
    observed = statistic_fn(left, right)
    states = _bounded_factorial(count, max_exact_states)
    selected = _select_method(method, states, max_exact_states)
    if selected == "exact" and states * count ** 2 > max_work:
        if method == "auto":
            selected = "asymptotic"
        else:
            raise ValueError(f"exact {test} enumeration exceeds max_resampling_work")
    if selected == "exact":
        extreme = sum(_extreme(statistic_fn(left, permutation), observed, 0, alternative)
                      for permutation in itertools.permutations(right))
        p_value = sp.Rational(extreme, states); method_name = "exact_permutation"
        trust = cap_trust(sample.input_trust, "exact"); numerical = None
    elif test == "spearman":
        rho = float(observed); degrees = count - 2
        if abs(rho) >= 1:
            probability = mp.mpf(0)
        else:
            t_value = rho * math.sqrt(degrees / ((1 - rho) * (1 + rho)))
            with mp.workdps(50):
                two_sided = mp.betainc(degrees / 2, .5, 0,
                                       degrees / (degrees + t_value ** 2),
                                       regularized=True)
            probability = two_sided if alternative == "two_sided" else (
                two_sided / 2 if (alternative == "greater") == (t_value >= 0)
                else 1 - two_sided / 2)
        p_value = _mp_expr(probability); method_name = "asymptotic_student_t"
        trust = cap_trust(sample.input_trust, "numeric_high_precision")
        numerical = "Student-t approximation for Spearman rho"
    else:
        _, score = _kendall_statistic(left, right)
        _, ties_left = _ranks(left); _, ties_right = _ranks(right)
        n = count
        variance_score = (n * (n - 1) * (2 * n + 5)
                          - sum(t * (t - 1) * (2 * t + 5) for t in ties_left)
                          - sum(t * (t - 1) * (2 * t + 5) for t in ties_right)) / 18
        if n > 2:
            variance_score += (
                sum(t * (t - 1) for t in ties_left) *
                sum(t * (t - 1) for t in ties_right) / (2 * n * (n - 1)) +
                sum(t * (t - 1) * (t - 2) for t in ties_left) *
                sum(t * (t - 1) * (t - 2) for t in ties_right) /
                (9 * n * (n - 1) * (n - 2)))
        if variance_score <= 0:
            raise ValueError("Kendall asymptotic variance is zero")
        z = score / math.sqrt(variance_score)
        p_value = _mp_expr(_normal_p(z, alternative)); method_name = "asymptotic_normal"
        trust = cap_trust(sample.input_trust, "numeric_high_precision")
        numerical = "tie-corrected normal approximation for Kendall tau-b"
    output = NonparametricTestResult(
        test=test, statistic=observed, p_value=p_value,
        alternative=alternative, method=method_name,
        variables=(left_name, right_name), sample_sizes=(count,),
        enumeration_count=states if selected == "exact" else None,
        configuration={"left": left_name, "right": right_name,
                       "alternative": alternative, "method": selected},
        input_trust=trust)
    checks = {"statistic_bounded": -1 <= observed <= 1,
              "p_value_bounded": 0 <= p_value <= 1,
              "paired_count_reconciled": len(left) == len(right)}
    return _result(sample, test, output, checks, method=method_name,
                   trust=trust, numerical=numerical)


def spearman(sample, left, right, **kwargs):
    return _association(sample, left, right, test="spearman", **kwargs)


def kendall(sample, left, right, **kwargs):
    return _association(sample, left, right, test="kendall", **kwargs)


def _location(values, statistic):
    if statistic == "difference_in_means" or statistic == "mean":
        return sp.simplify(sum(values, sp.S.Zero) / len(values))
    ordered = _sorted(tuple(values))
    return _quantile(ordered, 1, 2)


def permutation_test(sample: StatisticalSample, value: str, group: str,
                     group_a: sp.Expr, group_b: sp.Expr, *,
                     statistic="difference_in_means", alternative="two_sided",
                     method="auto", resamples=10_000, seed=None,
                     max_exact_states=100_000, max_work=20_000_000):
    if statistic not in {"difference_in_means", "difference_in_medians"}:
        raise ValueError("permutation statistic must be difference_in_means or difference_in_medians")
    if alternative not in ALTERNATIVES:
        raise ValueError("alternative must be two_sided, less, or greater")
    values = _column(sample, value); labels = _column(sample, group)
    left = tuple(item for item, label in zip(values, labels) if _equal(label, group_a))
    right = tuple(item for item, label in zip(values, labels) if _equal(label, group_b))
    if not left or not right or len(left) + len(right) != len(values):
        raise ValueError("group_a/group_b must partition the sample into nonempty groups")
    observed = sp.simplify(_location(left, statistic) - _location(right, statistic))
    pooled = left + right; n1 = len(left)
    states = _bounded_comb(len(pooled), n1, max_exact_states)
    if method not in {"auto", "exact", "monte_carlo"}:
        raise ValueError("method must be auto, exact, or monte_carlo")
    factor = max(1, (len(pooled) - 1).bit_length()) if statistic == "difference_in_medians" else 1
    exact_work = states * len(pooled) * factor
    selected = "exact" if method == "exact" or (
        method == "auto" and states <= max_exact_states and exact_work <= max_work
    ) else "monte_carlo"
    if selected == "exact" and states > max_exact_states:
        raise ValueError("exact enumeration exceeds max_exact_resampling_states")
    required_work = ((states if selected == "exact" else resamples) *
                     len(pooled) * factor)
    if required_work > max_work:
        raise ValueError("permutation_test exceeds max_resampling_work")
    if selected == "exact":
        extreme = 0; indices = set(range(len(pooled)))
        for chosen in itertools.combinations(range(len(pooled)), n1):
            chosen_set = set(chosen)
            candidate = _location(tuple(pooled[i] for i in chosen), statistic) - _location(
                tuple(pooled[i] for i in indices - chosen_set), statistic)
            extreme += _extreme(candidate, observed, 0, alternative)
        p_value = sp.Rational(extreme, states); completed = states
        trust = _input_trust(sample, group_a, group_b); simulation = None
        method_name = "exact_enumeration"
    else:
        if seed is None:
            raise ValueError("monte_carlo permutation requires an explicit seed")
        rng = np.random.Generator(np.random.PCG64(seed)); extreme = 0
        numeric = np.asarray([float(item) for item in pooled])
        observed_float = float(observed)
        for _ in range(resamples):
            shuffled = rng.permutation(numeric)
            candidate = (float(np.mean(shuffled[:n1]) - np.mean(shuffled[n1:]))
                         if statistic == "difference_in_means" else
                         float(np.median(shuffled[:n1]) - np.median(shuffled[n1:])))
            extreme += _extreme(candidate, observed_float, 0, alternative)
        p_value = sp.Rational(extreme + 1, resamples + 1); completed = resamples
        trust = "empirical"; method_name = "monte_carlo"
        mcse = math.sqrt(float(p_value) * (1 - float(p_value)) / (resamples + 1))
        simulation = {"seed": seed, "resamples": resamples,
                      "monte_carlo_standard_error": mcse}
    output = ResamplingResult(
        procedure="permutation_test", statistic_name=statistic,
        observed_statistic=observed, method=method_name,
        variables=(value, group), sample_sizes=(len(left), len(right)),
        p_value=p_value, resamples=completed, seed=seed if selected != "exact" else None,
        rng_algorithm="PCG64" if selected != "exact" else None,
        extreme_count=extreme,
        configuration={"value": value, "group": group, "group_a": group_a,
                       "group_b": group_b, "statistic": statistic,
                       "alternative": alternative, "method": selected,
                       "resamples": resamples, "seed": seed}, input_trust=trust)
    checks = {"group_sizes_reconciled": len(left) + len(right) == len(values),
              "p_value_bounded": 0 <= p_value <= 1,
              "draw_count_reconciled": completed == (states if selected == "exact" else resamples)}
    return _result(sample, "permutation_test", output, checks,
                   method=method_name, trust=trust, simulation=simulation)


def bootstrap(sample: StatisticalSample, variable: str, *, statistic="mean",
              confidence_level=.95, resamples=10_000, seed=None,
              batch_cells=1_000_000):
    if statistic not in {"mean", "median"}:
        raise ValueError("bootstrap statistic must be mean or median")
    if resamples < 2:
        raise ValueError("bootstrap requires at least two resamples")
    if seed is None:
        raise ValueError("bootstrap requires an explicit seed")
    values = _column(sample, variable); numeric = np.asarray(values, dtype=float)
    observed = _location(values, statistic)
    rng = np.random.Generator(np.random.PCG64(seed)); estimates = np.empty(resamples)
    batch = max(1, min(resamples, batch_cells // len(values)))
    for start in range(0, resamples, batch):
        stop = min(resamples, start + batch)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        draws = numeric[indices]
        estimates[start:stop] = (np.mean(draws, axis=1) if statistic == "mean"
                                 else np.median(draws, axis=1))
    alpha = (1 - confidence_level) / 2
    quantiles = np.quantile(estimates, [alpha, 1 - alpha], method="linear")
    interval = (sp.Float(repr(float(quantiles[0])), 15),
                sp.Float(repr(float(quantiles[1])), 15))
    standard_error = sp.Float(repr(float(np.std(estimates, ddof=1))), 15)
    level = sp.Float(repr(float(confidence_level)), 15)
    output = ResamplingResult(
        procedure="bootstrap", statistic_name=statistic,
        observed_statistic=observed, method="percentile_bootstrap",
        variables=(variable,), sample_sizes=(len(values),), interval=interval,
        standard_error=standard_error, confidence_level=level,
        resamples=resamples, seed=seed, rng_algorithm="PCG64",
        configuration={"variable": variable, "statistic": statistic,
                       "confidence_level": level, "resamples": resamples,
                       "seed": seed}, input_trust="empirical")
    checks = {"interval_ordered": interval[0] <= interval[1],
              "draw_count_reconciled": len(estimates) == resamples,
              "standard_error_nonnegative": standard_error >= 0,
              "seed_recorded": True}
    simulation = {"seed": seed, "resamples": resamples}
    result, output = _result(sample, "bootstrap", output, checks,
                             method="percentile_bootstrap", trust="empirical",
                             simulation=simulation)
    result.value.update({"interval": interval, "standard_error": standard_error,
                         "confidence_level": level})
    return result, output


def verify_nonparametric(sample: StatisticalSample, result: NonparametricTestResult,
                         max_exact_states: int, max_work: int, max_groups: int):
    config = dict(result.configuration); selected = config.pop("method")
    function = {
        "mann_whitney": mann_whitney, "wilcoxon": wilcoxon,
        "kruskal_wallis": kruskal_wallis, "ks_2samp": ks_2samp,
        "spearman": spearman, "kendall": kendall,
    }[result.test]
    if result.test in {"wilcoxon", "ks_2samp"}:
        left = config.pop("left"); right = config.pop("right")
        recomputed, candidate = function(
            sample, left, right, method=selected,
            max_exact_states=max_exact_states, max_work=max_work, **config)
    else:
        if result.test == "kruskal_wallis":
            config["max_groups"] = max_groups
        recomputed, candidate = function(sample, method=selected,
                                         max_exact_states=max_exact_states,
                                         max_work=max_work, **config)
    checks = {"statistic_recomputed": sp.simplify(candidate.statistic - result.statistic) == 0,
              "p_value_recomputed": sp.simplify(candidate.p_value - result.p_value) == 0,
              "method_reconciled": candidate.method == result.method}
    checks = {name: bool(accepted) for name, accepted in checks.items()}
    bundle = _bundle(sample, result.test, "deterministic_test_replay",
                     result.input_trust,
                     numerical="replay of asymptotic formula" if selected == "asymptotic" else None)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust, value={"checks": checks}, verification=checks,
        details={"population_generalization": "not_established",
                 "null_assumptions": "asserted_not_verified"},
        claim_evidence={"verify": bundle})


def verify_resampling(sample: StatisticalSample, result: ResamplingResult,
                      max_exact_states: int, batch_cells: int, max_work: int):
    config = dict(result.configuration)
    if result.procedure == "permutation_test":
        recomputed, candidate = permutation_test(
            sample, max_exact_states=max_exact_states, max_work=max_work, **config)
        checks = {"statistic_recomputed": sp.simplify(
                      candidate.observed_statistic - result.observed_statistic) == 0,
                  "p_value_replayed": candidate.p_value == result.p_value,
                  "extreme_count_replayed": candidate.extreme_count == result.extreme_count,
                  "rng_provenance_reconciled": candidate.seed == result.seed and
                                               candidate.rng_algorithm == result.rng_algorithm}
        simulation = None if result.method == "exact_enumeration" else {
            "seed": result.seed, "resamples": result.resamples}
    else:
        config["confidence_level"] = float(config["confidence_level"])
        recomputed, candidate = bootstrap(sample, batch_cells=batch_cells, **config)
        checks = {"statistic_recomputed": sp.simplify(
                      candidate.observed_statistic - result.observed_statistic) == 0,
                  "interval_replayed": candidate.interval == result.interval,
                  "standard_error_replayed": candidate.standard_error == result.standard_error,
                  "rng_provenance_reconciled": candidate.seed == result.seed and
                                               candidate.rng_algorithm == result.rng_algorithm}
        simulation = {"seed": result.seed, "resamples": result.resamples}
    checks = {name: bool(accepted) for name, accepted in checks.items()}
    bundle = _bundle(sample, result.procedure, "deterministic_resampling_replay",
                     result.input_trust, simulation=simulation)
    return EngineeringResult(
        operation="verify", status="verified" if all(checks.values()) else "refuted",
        trust=result.input_trust, value={"checks": checks}, verification=checks,
        details={"population_generalization": "not_established",
                 "resampling_assumptions": "asserted_not_verified"},
        claim_evidence={"verify": bundle})
