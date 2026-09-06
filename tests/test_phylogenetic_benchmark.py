# =============================================================================
# MathKernel - Tests for frozen phylogenetic benchmark protocols
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mathkernel.phylogenetic_benchmark import (
    AlignmentPartition,
    BenchmarkProtocol,
    DatasetManifest,
    FrozenBenchmarkManifest,
    QuartetCase,
    SequenceAlignment,
    SiteRange,
    audit_dataset,
    canonical_sha256,
    circular_block_bootstrap_indices,
    complete_nucleotide_quartet,
    count_quartet_states,
    filtered_partition_groups,
    infer_alignment_format,
    load_alignment,
    logdet_scores,
    parse_fasta,
    parse_newick,
    parse_nexus,
    parse_phylip,
    partition_block_bootstrap_indices,
    partition_stratified_bootstrap_indices,
    p_distance_scores,
    rank_tail_scores,
    run_quartet_benchmark,
    sample_reference_quartets,
    score_minimizers,
    sha256_file,
    site_bootstrap_indices,
    split_name,
    summarize_corpus,
)


def test_parse_fasta_and_complete_case_quartet() -> None:
    alignment = parse_fasta(
        ">a\nACGTN-\n>b\nACGTA-\n>c\nACGTAA\n>d\nACGTAA\n"
    )
    assert alignment.taxa == ("a", "b", "c", "d")
    assert alignment.n_sites == 6
    data = complete_nucleotide_quartet(alignment, ("a", "b", "c", "d"))
    assert data.states.shape == (4, 4)
    assert data.original_site_indices.tolist() == [0, 1, 2, 3]
    assert data.excluded_sites == 2
    assert data.states[:, 0].tolist() == [0, 0, 0, 0]


def test_parse_fasta_rejects_duplicates_and_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_fasta(">a\nAC\n>a\nAC\n")
    with pytest.raises(ValueError, match="equal length"):
        parse_fasta(">a\nAC\n>b\nA\n")


def test_parse_sequential_phylip() -> None:
    alignment = parse_phylip("4 6\na AACCGG\nb AACCGT\nc AATCGT\nd TATCGT\n")
    assert alignment.source_format == "PHYLIP"
    assert alignment.taxa == ("a", "b", "c", "d")
    assert alignment.sequences[3] == "TATCGT"


def test_parse_interleaved_phylip_with_unlabelled_continuation() -> None:
    alignment = parse_phylip(
        "4 8\n"
        "a AACC\n"
        "b AACT\n"
        "c AGCT\n"
        "d TGCT\n\n"
        "GGTT\n"
        "GGTA\n"
        "GGTA\n"
        "GGTA\n"
    )
    assert alignment.sequences == (
        "AACCGGTT",
        "AACTGGTA",
        "AGCTGGTA",
        "TGCTGGTA",
    )


def test_parse_interleaved_phylip_with_repeated_labels() -> None:
    alignment = parse_phylip(
        "4 8\n"
        "a AACC\n"
        "b AACT\n"
        "c AGCT\n"
        "d TGCT\n\n"
        "a GGTT\n"
        "b GGTA\n"
        "c GGTA\n"
        "d GGTA\n"
    )
    assert alignment.n_taxa == 4
    assert alignment.n_sites == 8


def test_parse_nexus_interleaved_matchchar_and_charsets() -> None:
    text = """#NEXUS
    [nested [comment] is ignored]
    begin data;
      dimensions ntax=4 nchar=8;
      format datatype=dna interleave gap=- missing=? matchchar=.;
      matrix
      'tax a' AACC
      b       AACT
      c       AGCT
      d       TGCT

      'tax a' GGTT
      b       GGTA
      c       ....
      d       GGTA
      ;
    end;
    begin sets;
      charset first = 1-4;
      charset codon_1 = 1-8\\3;
      charset tail = 5-.;
    end;
    """
    alignment = parse_nexus(text)
    assert alignment.taxa == ("tax a", "b", "c", "d")
    assert alignment.sequences[2] == "AGCTGGTT"
    assert [partition.name for partition in alignment.partitions] == [
        "first",
        "codon_1",
        "tail",
    ]
    assert alignment.partitions[1].site_indices(8).tolist() == [0, 3, 6]
    assert alignment.partitions[2].site_indices(8).tolist() == [4, 5, 6, 7]


def test_alignment_format_inference(tmp_path: Path) -> None:
    assert infer_alignment_format(tmp_path / "x.fa") == "FASTA"
    assert infer_alignment_format(tmp_path / "x.nex") == "NEXUS"
    assert infer_alignment_format(tmp_path / "x.phy") == "PHYLIP"
    assert infer_alignment_format(tmp_path / "x.unknown", ">a\nAC\n") == "FASTA"


def test_newick_parser_handles_quotes_lengths_and_comments() -> None:
    tree = parse_newick("(('tax a':0.1,b:0.2)[x]:0.3,(c,d)95:0.4)root;")
    assert set(tree.leaf_labels) == {"tax a", "b", "c", "d"}
    assert tree.induced_quartet_split(("tax a", "b", "c", "d")) == (0, 1)
    assert tree.topological_distance("tax a", "b") == 2
    assert tree.topological_distance("tax a", "c") > 2


def test_newick_unresolved_quartet_returns_none() -> None:
    tree = parse_newick("(a,b,c,d);")
    assert tree.induced_quartet_split(("a", "b", "c", "d")) is None


def test_newick_induced_split_is_coordinate_aware() -> None:
    tree = parse_newick("((a,b),(c,d));")
    assert tree.induced_quartet_split(("c", "a", "d", "b")) == (0, 2)
    assert split_name((0, 2)) == "02|13"


def test_informative_edges_deduplicate_degree_two_root_split() -> None:
    tree = parse_newick("((a,b),(c,d));")
    edges = tree.informative_edges()
    assert len(edges) == 1
    left, right = tree.edge_leaf_split(*edges[0])
    assert {frozenset(left), frozenset(right)} == {
        frozenset(("a", "b")),
        frozenset(("c", "d")),
    }


def test_reference_quartet_sampling_is_reproducible_and_result_blind() -> None:
    tree = parse_newick("(((a,b),(c,d)),((e,f),(g,h)));")
    first = sample_reference_quartets(
        tree,
        "demo",
        seed=11,
        quartets_per_edge=2,
        maximum=8,
    )
    second = sample_reference_quartets(
        tree,
        "demo",
        seed=11,
        quartets_per_edge=2,
        maximum=8,
    )
    third = sample_reference_quartets(
        tree,
        "demo",
        seed=12,
        quartets_per_edge=2,
        maximum=8,
    )
    assert first == second
    assert 1 <= len(first) <= 8
    assert all(
        case.reference_split == tree.induced_quartet_split(case.taxa)
        for case in first
    )
    assert first != third
    assert any(case.reference_split != (0, 1) for case in first)


def test_site_and_block_bootstraps_are_reproducible() -> None:
    first = site_bootstrap_indices(20, np.random.default_rng(3))
    second = site_bootstrap_indices(20, np.random.default_rng(3))
    assert np.array_equal(first, second)
    assert first.shape == (20,)

    block = circular_block_bootstrap_indices(10, 4, np.random.default_rng(9))
    assert block.shape == (10,)
    for start in range(0, 8, 4):
        assert np.all((np.diff(block[start : start + 4]) % 10) == 1)


def test_filtered_partition_groups_are_disjoint_under_overlap() -> None:
    alignment = SequenceAlignment(
        taxa=("a", "b", "c", "d"),
        sequences=("ACGTACGT",) * 4,
        partitions=(
            AlignmentPartition("first", (SiteRange(0, 4),)),
            AlignmentPartition("odd", (SiteRange(0, 8, 2),)),
        ),
    )
    data = complete_nucleotide_quartet(alignment, alignment.taxa)
    groups = filtered_partition_groups(alignment, data)
    assert set(groups) == {"first", "first&odd", "odd", "__unassigned__"}
    concatenated = np.concatenate(list(groups.values()))
    assert sorted(concatenated.tolist()) == list(range(8))
    assert len(set(concatenated.tolist())) == 8

    stratified = partition_stratified_bootstrap_indices(
        groups,
        np.random.default_rng(1),
    )
    assert stratified.size == 8
    block = partition_block_bootstrap_indices(groups, np.random.default_rng(4))
    assert block.size > 0


def _strong_quartet_states(repetitions: int = 250) -> np.ndarray:
    patterns: list[tuple[int, int, int, int]] = []
    for ancestral in range(4):
        for right in range(4):
            patterns.extend([(ancestral, ancestral, right, right)] * repetitions)
            patterns.extend(
                [((ancestral + 1) % 4, ancestral, right, right)]
                * max(1, repetitions // 20)
            )
            patterns.extend(
                [(ancestral, ancestral, (right + 1) % 4, right)]
                * max(1, repetitions // 20)
            )
    return np.asarray(patterns, dtype=np.int8).T


def test_quartet_scores_and_tie_safe_minimizers() -> None:
    states = _strong_quartet_states()
    assert score_minimizers(rank_tail_scores(states)) == ("01|23",)
    assert score_minimizers(p_distance_scores(states)) == ("01|23",)
    assert score_minimizers(logdet_scores(states, pseudocount=0.5)) == ("01|23",)
    tied = score_minimizers(
        {"01|23": 1.0, "02|13": 1.0, "03|12": 2.0}
    )
    assert tied == ("01|23", "02|13")


def test_count_tensor_preserves_total() -> None:
    states = _strong_quartet_states(10)
    tensor = count_quartet_states(states)
    assert tensor.shape == (4, 4, 4, 4)
    assert tensor.sum() == states.shape[1]


def test_protocol_and_manifest_hashes_are_canonical(tmp_path: Path) -> None:
    alignment_path = tmp_path / "a.fasta"
    tree_path = tmp_path / "a.nwk"
    alignment_path.write_text(
        ">a\nACGT\n>b\nACGT\n>c\nTGCA\n>d\nTGCA\n",
        encoding="utf-8",
    )
    tree_path.write_text("((a,b),(c,d));\n", encoding="utf-8")
    dataset = DatasetManifest.from_files("demo", alignment_path, tree_path)
    protocol = BenchmarkProtocol(
        bootstrap_replicates=3,
        resampling_modes=("site",),
    )
    manifest = FrozenBenchmarkManifest(protocol, (dataset,))
    assert len(protocol.digest) == 64
    assert len(manifest.digest) == 64
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )
    output = tmp_path / "manifest.json"
    manifest.write(output)
    assert manifest.digest in output.read_text(encoding="utf-8")
    assert dataset.verify() == (True, ())
    alignment_path.write_text(
        alignment_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert dataset.verify()[1] == ("alignment_hash_mismatch",)


def test_dataset_manifest_supports_paths_relative_to_a_frozen_root(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    data = corpus / "data"
    data.mkdir(parents=True)
    alignment_path = data / "demo.fasta"
    tree_path = data / "demo.nwk"
    alignment_path.write_text(
        ">a\nAAAA\n>b\nAAAT\n>c\nAATT\n>d\nTTTT\n",
        encoding="utf-8",
    )
    tree_path.write_text("((a,b),(c,d));\n", encoding="utf-8")
    dataset = DatasetManifest.from_files(
        "demo",
        alignment_path,
        tree_path,
        path_root=corpus,
    )
    assert dataset.alignment_path == "data/demo.fasta"
    assert dataset.tree_path == "data/demo.nwk"
    assert dataset.verify(base_dir=corpus) == (True, ())
    audit, alignment, tree, cases = audit_dataset(
        dataset,
        BenchmarkProtocol(minimum_complete_sites=1),
        base_dir=corpus,
    )
    assert audit.eligible
    assert alignment is not None and tree is not None and len(cases) == 1


def test_load_alignment_records_source_hash(tmp_path: Path) -> None:
    path = tmp_path / "example.fasta"
    path.write_text(">a\nAC\n>b\nAC\n", encoding="utf-8")
    alignment = load_alignment(path)
    assert alignment.metadata["sha256"] == sha256_file(path)


def test_dataset_audit_and_quartet_benchmark(tmp_path: Path) -> None:
    states = _strong_quartet_states(20)
    symbols = np.asarray(list("ACGT"))
    alignment_path = tmp_path / "demo.fasta"
    with alignment_path.open("w", encoding="utf-8") as handle:
        for index, label in enumerate(("a", "b", "c", "d")):
            handle.write(f">{label}\n{''.join(symbols[states[index]])}\n")
    tree_path = tmp_path / "demo.nwk"
    tree_path.write_text("((a,b),(c,d));\n", encoding="utf-8")
    dataset = DatasetManifest.from_files("demo", alignment_path, tree_path)
    protocol = BenchmarkProtocol(
        seed=44,
        max_quartets_per_dataset=4,
        quartets_per_edge=1,
        minimum_complete_sites=50,
        methods=("rank_tail", "p_distance"),
        resampling_modes=("site", "circular_block"),
        bootstrap_replicates=20,
        circular_block_length=8,
    )
    audit, alignment, tree, cases = audit_dataset(dataset, protocol)
    assert audit.eligible
    assert audit.sampled_quartets == 1
    assert alignment is not None and tree is not None
    result = run_quartet_benchmark(alignment, cases[0], protocol)
    assert result.eligible
    assert len(result.bootstrap_summaries) == 4
    assert all(summary.strict_support > 0.8 for summary in result.bootstrap_summaries)
    corpus = summarize_corpus((result,))
    assert corpus["eligible_cases"] == 1
    assert corpus["full_data"]["rank_tail"]["unique_accuracy"] == 1.0


def test_benchmark_rejects_too_few_complete_sites() -> None:
    alignment = parse_fasta(">a\nANNN\n>b\nANNN\n>c\nTNNN\n>d\nTNNN\n")
    case = QuartetCase("demo", ("a", "b", "c", "d"), (0, 1))
    protocol = BenchmarkProtocol(
        minimum_complete_sites=2,
        bootstrap_replicates=0,
        resampling_modes=(),
    )
    result = run_quartet_benchmark(alignment, case, protocol)
    assert not result.eligible
    assert "only 1 complete sites" in (result.rejection_reason or "")


def test_manifest_audit_fails_closed_on_hash_mismatch(tmp_path: Path) -> None:
    alignment_path = tmp_path / "a.fa"
    tree_path = tmp_path / "a.nwk"
    alignment_path.write_text(
        ">a\nAC\n>b\nAC\n>c\nGT\n>d\nGT\n",
        encoding="utf-8",
    )
    tree_path.write_text("((a,b),(c,d));", encoding="utf-8")
    dataset = DatasetManifest.from_files("demo", alignment_path, tree_path)
    alignment_path.write_text(
        ">a\nAA\n>b\nAA\n>c\nGG\n>d\nGG\n",
        encoding="utf-8",
    )
    audit, alignment, tree, cases = audit_dataset(dataset, BenchmarkProtocol())
    assert not audit.eligible
    assert audit.manifest_problems == ("alignment_hash_mismatch",)
    assert alignment is None and tree is None and cases == ()


def test_protocol_validation() -> None:
    with pytest.raises(ValueError, match="unknown benchmark methods"):
        BenchmarkProtocol(methods=("magic",))
    with pytest.raises(ValueError, match="unknown resampling modes"):
        BenchmarkProtocol(resampling_modes=("magic",))
    with pytest.raises(ValueError, match="maximum_missing_fraction"):
        BenchmarkProtocol(maximum_missing_fraction=1.0)
