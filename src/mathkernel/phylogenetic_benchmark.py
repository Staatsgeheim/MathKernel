# =============================================================================
# MathKernel - Reproducible phylogenetic benchmark protocols and ingestion
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Frozen, reference-tree-aware benchmarking for quartet diagnostics.

The mathematical phylogenetics modules answer exact and statistical questions
for a single model or quartet.  This module supplies the protocol layer needed
for honest multi-dataset validation:

* FASTA, relaxed PHYLIP, practical NEXUS, and Newick ingestion;
* SHA-256 provenance and canonical protocol/manifest locks;
* deterministic, result-blind quartet sampling from reference-tree splits;
* explicit complete-case nucleotide filtering with source-site provenance;
* site, circular-block, partition-stratified, and whole-partition resampling;
* ordinary rank-tail, p-distance, and normalized log-det baselines; and
* tie-safe, failure-aware result summaries.

Reference trees are inputs, not inferred from the alignment being scored.  The
module therefore separates corpus definition from execution and makes silent
post-hoc dataset or quartet selection difficult.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .phylogenetic_inference import rank_tail_frobenius


NUCLEOTIDE_ALPHABET = "ACGT"
NUCLEOTIDE_CODE: Mapping[str, int] = {
    symbol: index for index, symbol in enumerate(NUCLEOTIDE_ALPHABET)
}
CANONICAL_QUARTET_SPLITS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
)


# ---------------------------------------------------------------------------
# Canonical serialization and provenance
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Hash a file without reading it completely into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return stable JSON used for protocol and corpus digests."""

    return json.dumps(
        _json_ready(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


# ---------------------------------------------------------------------------
# Alignment representation and parsers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SiteRange:
    """Zero-based, stop-exclusive arithmetic site range."""

    start: int
    stop: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < 0:
            raise ValueError("site range endpoints must be nonnegative")
        if self.stop <= self.start:
            raise ValueError("site range stop must exceed start")
        if self.step <= 0:
            raise ValueError("site range step must be positive")

    def indices(self, length: int | None = None) -> range:
        stop = self.stop if length is None else min(self.stop, int(length))
        if self.start >= stop:
            return range(0)
        return range(self.start, stop, self.step)


@dataclass(frozen=True, slots=True)
class AlignmentPartition:
    """Named collection of one or more site ranges."""

    name: str
    ranges: tuple[SiteRange, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("partition name must be nonempty")
        if not self.ranges:
            raise ValueError("partition must contain at least one site range")

    def site_indices(self, length: int) -> np.ndarray:
        sites: set[int] = set()
        for item in self.ranges:
            sites.update(item.indices(length))
        return np.asarray(sorted(sites), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SequenceAlignment:
    """An aligned character matrix with preserved taxon labels."""

    taxa: tuple[str, ...]
    sequences: tuple[str, ...]
    datatype: str = "DNA"
    source_format: str = "FASTA"
    gap: str = "-"
    missing: str = "?"
    matchchar: str | None = None
    partitions: tuple[AlignmentPartition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.taxa:
            raise ValueError("alignment must contain at least one taxon")
        if len(self.taxa) != len(self.sequences):
            raise ValueError("taxa and sequences must have equal length")
        if len(set(self.taxa)) != len(self.taxa):
            raise ValueError("alignment taxon labels must be unique")
        if any(not label for label in self.taxa):
            raise ValueError("alignment taxon labels must be nonempty")
        lengths = {len(sequence) for sequence in self.sequences}
        if len(lengths) != 1:
            raise ValueError("all aligned sequences must have equal length")
        if next(iter(lengths)) <= 0:
            raise ValueError("alignment sequences must be nonempty")
        if any(any(character.isspace() for character in sequence) for sequence in self.sequences):
            raise ValueError("alignment sequences may not contain whitespace")
        for partition in self.partitions:
            if np.any(partition.site_indices(self.n_sites) >= self.n_sites):
                raise ValueError("partition site out of alignment bounds")

    @property
    def n_taxa(self) -> int:
        return len(self.taxa)

    @property
    def n_sites(self) -> int:
        return len(self.sequences[0])

    def sequence_map(self) -> Mapping[str, str]:
        return dict(zip(self.taxa, self.sequences, strict=True))

    def subset(self, taxa: Sequence[str]) -> "SequenceAlignment":
        selected = tuple(taxa)
        mapping = self.sequence_map()
        missing = tuple(label for label in selected if label not in mapping)
        if missing:
            raise KeyError(f"alignment does not contain taxa: {missing}")
        return SequenceAlignment(
            taxa=selected,
            sequences=tuple(mapping[label] for label in selected),
            datatype=self.datatype,
            source_format=self.source_format,
            gap=self.gap,
            missing=self.missing,
            matchchar=self.matchchar,
            partitions=self.partitions,
            metadata=self.metadata,
        )

    def character_matrix(self, taxa: Sequence[str] | None = None) -> np.ndarray:
        selected = self if taxa is None else self.subset(taxa)
        return np.asarray([list(sequence) for sequence in selected.sequences], dtype="U1")


@dataclass(frozen=True, slots=True)
class QuartetSiteData:
    taxa: tuple[str, str, str, str]
    states: np.ndarray
    original_site_indices: np.ndarray
    total_alignment_sites: int
    excluded_sites: int

    def __post_init__(self) -> None:
        states = np.asarray(self.states)
        indices = np.asarray(self.original_site_indices)
        if states.ndim != 2 or states.shape[0] != 4:
            raise ValueError("quartet states must have shape (4, n_sites)")
        if indices.ndim != 1 or indices.size != states.shape[1]:
            raise ValueError("site indices must match quartet state columns")
        if np.any(states < 0) or np.any(states > 3):
            raise ValueError("quartet states must use nucleotide codes 0..3")
        if len(set(self.taxa)) != 4:
            raise ValueError("quartet taxa must be distinct")
        if self.total_alignment_sites < states.shape[1]:
            raise ValueError("total alignment sites cannot be smaller than retained sites")
        if self.excluded_sites != self.total_alignment_sites - states.shape[1]:
            raise ValueError("excluded-site count is inconsistent")

    @property
    def n_sites(self) -> int:
        return int(self.states.shape[1])

    @property
    def complete_fraction(self) -> float:
        return self.n_sites / self.total_alignment_sites


_LABEL_FRAGMENT = re.compile(r"^\s*(?:'((?:[^']|'')*)'|([^\s]+))\s+(.+?)\s*$")


def _parse_labeled_fragment(line: str) -> tuple[str, str]:
    match = _LABEL_FRAGMENT.match(line)
    if not match:
        raise ValueError(f"expected taxon label followed by sequence fragment: {line!r}")
    quoted, bare, fragment = match.groups()
    label = quoted.replace("''", "'") if quoted is not None else str(bare)
    sequence = "".join(fragment.split()).upper()
    if not sequence:
        raise ValueError(f"empty sequence fragment for taxon {label!r}")
    return label, sequence


def parse_fasta(text: str, *, metadata: Mapping[str, Any] | None = None) -> SequenceAlignment:
    taxa: list[str] = []
    sequences: list[str] = []
    current: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            label = line[1:].strip().split()[0]
            if not label:
                raise ValueError("FASTA identifier must be nonempty")
            if label in taxa:
                raise ValueError(f"duplicate FASTA identifier {label!r}")
            taxa.append(label)
            sequences.append("")
            current = len(taxa) - 1
        else:
            if current is None:
                raise ValueError("FASTA sequence data appeared before an identifier")
            sequences[current] += "".join(line.split()).upper()
    if not taxa:
        raise ValueError("FASTA alignment contains no records")
    return SequenceAlignment(
        taxa=tuple(taxa),
        sequences=tuple(sequences),
        source_format="FASTA",
        metadata={} if metadata is None else dict(metadata),
    )


def _first_nonempty_line(lines: Sequence[str]) -> tuple[int, str]:
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line:
            return index, line
    raise ValueError("alignment text is empty")


def parse_phylip(text: str, *, metadata: Mapping[str, Any] | None = None) -> SequenceAlignment:
    """Parse common relaxed sequential and interleaved PHYLIP files."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    header_index, header = _first_nonempty_line(lines)
    parts = header.split()
    if len(parts) < 2:
        raise ValueError("PHYLIP header must contain ntax and nchar")
    try:
        ntax, nchar = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError("invalid PHYLIP ntax/nchar header") from exc
    if ntax <= 0 or nchar <= 0:
        raise ValueError("PHYLIP ntax and nchar must be positive")

    payload = lines[header_index + 1 :]
    taxa: list[str] = []
    fragments: dict[str, list[str]] = {}
    cursor = 0
    while cursor < len(payload) and len(taxa) < ntax:
        line = payload[cursor].strip()
        cursor += 1
        if not line:
            continue
        label, fragment = _parse_labeled_fragment(line)
        if label in fragments:
            raise ValueError("first PHYLIP block contains a duplicate taxon")
        taxa.append(label)
        fragments[label] = [fragment]
    if len(taxa) != ntax:
        raise ValueError(f"PHYLIP declared {ntax} taxa but first block supplied {len(taxa)}")

    continuation_order = 0
    while any(sum(map(len, fragments[label])) < nchar for label in taxa):
        while cursor < len(payload) and not payload[cursor].strip():
            cursor += 1
        if cursor >= len(payload):
            break
        line = payload[cursor].strip()
        cursor += 1
        match = _LABEL_FRAGMENT.match(line)
        if match:
            quoted, bare, raw_fragment = match.groups()
            possible_label = quoted.replace("''", "'") if quoted is not None else str(bare)
            if possible_label in fragments:
                fragments[possible_label].append("".join(raw_fragment.split()).upper())
                continuation_order = (taxa.index(possible_label) + 1) % ntax
                continue
        fragment = "".join(line.split()).upper()
        if fragment:
            label = taxa[continuation_order]
            fragments[label].append(fragment)
            continuation_order = (continuation_order + 1) % ntax

    sequences = tuple("".join(fragments[label]) for label in taxa)
    lengths = tuple(len(sequence) for sequence in sequences)
    if any(length != nchar for length in lengths):
        raise ValueError(f"PHYLIP declared nchar={nchar}, reconstructed lengths={lengths}")
    return SequenceAlignment(
        taxa=tuple(taxa),
        sequences=sequences,
        source_format="PHYLIP",
        metadata={} if metadata is None else dict(metadata),
    )


def _strip_nexus_comments(text: str) -> str:
    result: list[str] = []
    depth = 0
    index = 0
    in_quote = False
    while index < len(text):
        character = text[index]
        if character == "'" and depth == 0:
            if in_quote and index + 1 < len(text) and text[index + 1] == "'":
                result.extend(("'", "'"))
                index += 2
                continue
            in_quote = not in_quote
            result.append(character)
        elif not in_quote and character == "[":
            depth += 1
        elif not in_quote and character == "]" and depth:
            depth -= 1
        elif depth == 0:
            result.append(character)
        index += 1
    if depth:
        raise ValueError("unterminated NEXUS comment")
    return "".join(result)


def _parse_nexus_ranges(expression: str, nchar: int) -> tuple[SiteRange, ...]:
    ranges: list[SiteRange] = []
    for token in re.split(r"[\s,]+", expression.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+|\.))?(?:\\(\d+))?", token)
        if not match:
            raise ValueError(f"unsupported NEXUS site range {token!r}")
        start_raw, stop_raw, step_raw = match.groups()
        start = int(start_raw) - 1
        stop_inclusive = nchar if stop_raw == "." else int(stop_raw or start_raw)
        step = int(step_raw or 1)
        if start < 0 or stop_inclusive <= start or stop_inclusive > nchar:
            raise ValueError(f"NEXUS site range {token!r} is out of bounds for nchar={nchar}")
        ranges.append(SiteRange(start, stop_inclusive, step))
    if not ranges:
        raise ValueError("empty NEXUS charset")
    return tuple(ranges)


def parse_nexus(text: str, *, metadata: Mapping[str, Any] | None = None) -> SequenceAlignment:
    clean = _strip_nexus_comments(text).replace("\r\n", "\n").replace("\r", "\n")
    dimensions = re.search(r"\bdimensions\b([^;]*);", clean, flags=re.IGNORECASE | re.DOTALL)
    if not dimensions:
        raise ValueError("NEXUS data block is missing DIMENSIONS")
    ntax_match = re.search(r"\bntax\s*=\s*(\d+)", dimensions.group(1), flags=re.IGNORECASE)
    nchar_match = re.search(r"\bnchar\s*=\s*(\d+)", dimensions.group(1), flags=re.IGNORECASE)
    if not ntax_match or not nchar_match:
        raise ValueError("NEXUS DIMENSIONS must declare ntax and nchar")
    ntax, nchar = int(ntax_match.group(1)), int(nchar_match.group(1))

    format_match = re.search(r"\bformat\b([^;]*);", clean, flags=re.IGNORECASE | re.DOTALL)
    format_text = format_match.group(1) if format_match else ""
    datatype_match = re.search(r"\bdatatype\s*=\s*([^\s;]+)", format_text, flags=re.IGNORECASE)
    gap_match = re.search(r"\bgap\s*=\s*([^\s;]+)", format_text, flags=re.IGNORECASE)
    missing_match = re.search(r"\bmissing\s*=\s*([^\s;]+)", format_text, flags=re.IGNORECASE)
    matchchar_match = re.search(r"\bmatchchar\s*=\s*([^\s;]+)", format_text, flags=re.IGNORECASE)
    datatype = (datatype_match.group(1) if datatype_match else "DNA").upper()
    gap = gap_match.group(1)[0] if gap_match else "-"
    missing = missing_match.group(1)[0] if missing_match else "?"
    matchchar = matchchar_match.group(1)[0] if matchchar_match else None

    matrix_match = re.search(r"\bmatrix\b(.*?);", clean, flags=re.IGNORECASE | re.DOTALL)
    if not matrix_match:
        raise ValueError("NEXUS data block is missing MATRIX")
    fragments: dict[str, list[str]] = {}
    taxa: list[str] = []
    continuation_order = 0
    for raw in matrix_match.group(1).splitlines():
        line = raw.strip()
        if not line:
            continuation_order = 0
            continue
        match = _LABEL_FRAGMENT.match(line)
        if match:
            quoted, bare, raw_fragment = match.groups()
            label = quoted.replace("''", "'") if quoted is not None else str(bare)
            fragment = "".join(raw_fragment.split()).upper()
            if label not in fragments:
                if len(taxa) >= ntax:
                    # Once every taxon is known, a line that does not repeat a
                    # known label is an unlabelled continuation fragment.
                    label = taxa[continuation_order]
                    fragment = "".join(line.split()).upper()
                    continuation_order = (continuation_order + 1) % ntax
                else:
                    taxa.append(label)
                    fragments[label] = []
            else:
                continuation_order = (taxa.index(label) + 1) % ntax
            fragments[label].append(fragment)
        else:
            if len(taxa) != ntax:
                raise ValueError("unlabelled NEXUS continuation before all taxa were declared")
            label = taxa[continuation_order]
            fragments[label].append("".join(line.split()).upper())
            continuation_order = (continuation_order + 1) % ntax
    if len(taxa) != ntax:
        raise ValueError(f"NEXUS declared ntax={ntax}, parsed {len(taxa)} taxa")
    sequences = ["".join(fragments[label]) for label in taxa]
    if matchchar is not None and sequences:
        reference = sequences[0]
        replaced = [reference]
        for sequence in sequences[1:]:
            if len(sequence) != len(reference):
                raise ValueError("MATCHCHAR expansion requires equal preliminary lengths")
            replaced.append(
                "".join(
                    reference[index] if character == matchchar else character
                    for index, character in enumerate(sequence)
                )
            )
        sequences = replaced
    lengths = tuple(len(sequence) for sequence in sequences)
    if any(length != nchar for length in lengths):
        raise ValueError(f"NEXUS declared nchar={nchar}, reconstructed lengths={lengths}")

    partitions: list[AlignmentPartition] = []
    for match in re.finditer(
        r"\bcharset\s+(?:'((?:[^']|'')*)'|([^\s=]+))\s*=\s*([^;]+);",
        clean,
        flags=re.IGNORECASE,
    ):
        quoted, bare, expression = match.groups()
        name = quoted.replace("''", "'") if quoted is not None else str(bare)
        partitions.append(AlignmentPartition(name, _parse_nexus_ranges(expression, nchar)))

    return SequenceAlignment(
        taxa=tuple(taxa),
        sequences=tuple(sequences),
        datatype=datatype,
        source_format="NEXUS",
        gap=gap,
        missing=missing,
        matchchar=matchchar,
        partitions=tuple(partitions),
        metadata={} if metadata is None else dict(metadata),
    )


def infer_alignment_format(path: str | Path, text: str | None = None) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".fa", ".fas", ".fasta", ".fna", ".faa"}:
        return "FASTA"
    if suffix in {".nex", ".nexus", ".nxs"}:
        return "NEXUS"
    if suffix in {".phy", ".phylip"}:
        return "PHYLIP"
    if text is not None:
        stripped = text.lstrip()
        if stripped.upper().startswith("#NEXUS"):
            return "NEXUS"
        if stripped.startswith(">"):
            return "FASTA"
        if re.match(r"\d+\s+\d+", stripped):
            return "PHYLIP"
    raise ValueError(f"cannot infer alignment format for {path}")


def load_alignment(
    path: str | Path,
    *,
    format: str | None = None,
    encoding: str = "utf-8",
    metadata: Mapping[str, Any] | None = None,
) -> SequenceAlignment:
    source = Path(path)
    text = source.read_text(encoding=encoding)
    selected = (format or infer_alignment_format(source, text)).upper()
    source_metadata = {
        "path": str(source),
        "sha256": sha256_file(source),
        **({} if metadata is None else dict(metadata)),
    }
    if selected == "FASTA":
        return parse_fasta(text, metadata=source_metadata)
    if selected == "PHYLIP":
        return parse_phylip(text, metadata=source_metadata)
    if selected == "NEXUS":
        return parse_nexus(text, metadata=source_metadata)
    raise ValueError(f"unsupported alignment format {selected!r}")


def complete_nucleotide_quartet(
    alignment: SequenceAlignment,
    taxa: Sequence[str],
    *,
    allowed: str = NUCLEOTIDE_ALPHABET,
) -> QuartetSiteData:
    selected = tuple(taxa)
    if len(selected) != 4 or len(set(selected)) != 4:
        raise ValueError("exactly four distinct taxa are required")
    if len(set(allowed)) != 4:
        raise ValueError("quartet nucleotide alphabet must contain four unique symbols")
    matrix = alignment.character_matrix(selected)
    allowed_array = np.asarray(tuple(allowed), dtype="U1")
    valid = np.all(np.isin(matrix, allowed_array), axis=0)
    filtered = matrix[:, valid]
    mapping = {symbol: index for index, symbol in enumerate(allowed)}
    states = np.asarray(
        [[mapping[character] for character in row] for row in filtered],
        dtype=np.int8,
    )
    return QuartetSiteData(
        taxa=selected,  # type: ignore[arg-type]
        states=states,
        original_site_indices=np.flatnonzero(valid).astype(np.int64),
        total_alignment_sites=alignment.n_sites,
        excluded_sites=int((~valid).sum()),
    )


# ---------------------------------------------------------------------------
# Dependency-free Newick tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewickNode:
    id: int
    label: str | None
    parent: int | None
    children: tuple[int, ...]
    branch_length: float | None


@dataclass(frozen=True, slots=True)
class NewickTree:
    nodes: tuple[NewickNode, ...]
    root: int
    source: str | None = None

    def __post_init__(self) -> None:
        ids = {node.id for node in self.nodes}
        if ids != set(range(len(self.nodes))):
            raise ValueError("Newick node identifiers must be contiguous")
        if self.root not in ids:
            raise ValueError("Newick root is absent")
        leaf_labels = [node.label for node in self.nodes if not node.children]
        if any(label is None or label == "" for label in leaf_labels):
            raise ValueError("every Newick leaf must have a label")
        if len(set(leaf_labels)) != len(leaf_labels):
            raise ValueError("Newick leaf labels must be unique")

    @property
    def leaf_labels(self) -> tuple[str, ...]:
        return tuple(
            node.label
            for node in self.nodes
            if not node.children and node.label is not None
        )

    def adjacency(self) -> tuple[tuple[int, ...], ...]:
        neighbors: list[list[int]] = [[] for _ in self.nodes]
        for node in self.nodes:
            if node.parent is not None:
                neighbors[node.id].append(node.parent)
                neighbors[node.parent].append(node.id)
        return tuple(tuple(items) for items in neighbors)

    def leaf_id(self, label: str) -> int:
        for node in self.nodes:
            if not node.children and node.label == label:
                return node.id
        raise KeyError(f"tree does not contain leaf {label!r}")

    def _selected_leaves_on_side(
        self,
        start: int,
        blocked: int,
        selected: set[str],
    ) -> frozenset[str]:
        adjacency = self.adjacency()
        stack = [start]
        seen = {blocked}
        found: set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes[node_id]
            if not node.children and node.label in selected:
                found.add(str(node.label))
            stack.extend(neighbor for neighbor in adjacency[node_id] if neighbor not in seen)
        return frozenset(found)

    def edge_leaf_split(self, first: int, second: int) -> tuple[frozenset[str], frozenset[str]]:
        all_leaves = frozenset(self.leaf_labels)
        left = self._selected_leaves_on_side(first, second, set(all_leaves))
        return left, all_leaves - left

    def informative_edges(self, taxa: Iterable[str] | None = None) -> tuple[tuple[int, int], ...]:
        selected = set(self.leaf_labels if taxa is None else taxa)
        if not selected.issubset(set(self.leaf_labels)):
            raise KeyError("selected taxa are not all present in the tree")
        result: list[tuple[int, int]] = []
        seen_splits: set[frozenset[frozenset[str]]] = set()
        for node in self.nodes:
            if node.parent is None:
                continue
            side = self._selected_leaves_on_side(node.id, node.parent, selected)
            other = frozenset(selected - set(side))
            if len(side) >= 2 and len(other) >= 2:
                split_key = frozenset((side, other))
                if split_key in seen_splits:
                    continue
                seen_splits.add(split_key)
                result.append((node.id, node.parent))
        return tuple(result)

    def induced_quartet_split(self, taxa: Sequence[str]) -> tuple[int, int] | None:
        selected = tuple(taxa)
        if len(selected) != 4 or len(set(selected)) != 4:
            raise ValueError("an induced quartet requires four distinct taxa")
        selected_set = set(selected)
        missing = selected_set - set(self.leaf_labels)
        if missing:
            raise KeyError(f"tree does not contain quartet taxa: {sorted(missing)}")
        candidates: set[frozenset[frozenset[str]]] = set()
        for node in self.nodes:
            if node.parent is None:
                continue
            side = self._selected_leaves_on_side(node.id, node.parent, selected_set)
            if len(side) == 2:
                other = frozenset(selected_set - set(side))
                candidates.add(frozenset((side, other)))
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError("tree yielded incompatible quartet splits")
        partition = next(iter(candidates))
        side_with_zero = next(side for side in partition if selected[0] in side)
        indices = tuple(sorted(selected.index(label) for label in side_with_zero))
        return indices  # type: ignore[return-value]

    def topological_distance(self, first_label: str, second_label: str) -> int:
        first, second = self.leaf_id(first_label), self.leaf_id(second_label)
        adjacency = self.adjacency()
        queue: list[tuple[int, int]] = [(first, 0)]
        seen = {first}
        cursor = 0
        while cursor < len(queue):
            node_id, distance = queue[cursor]
            cursor += 1
            if node_id == second:
                return distance
            for neighbor in adjacency[node_id]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, distance + 1))
        raise ValueError("tree is disconnected")


class _NewickParser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0
        self.mutable: list[dict[str, Any]] = []

    def _skip(self) -> None:
        while self.position < len(self.text):
            character = self.text[self.position]
            if character.isspace():
                self.position += 1
                continue
            if character == "[":
                depth = 1
                self.position += 1
                while self.position < len(self.text) and depth:
                    if self.text[self.position] == "[":
                        depth += 1
                    elif self.text[self.position] == "]":
                        depth -= 1
                    self.position += 1
                if depth:
                    raise ValueError("unterminated Newick comment")
                continue
            break

    def _peek(self) -> str:
        self._skip()
        return self.text[self.position] if self.position < len(self.text) else ""

    def _label(self) -> str | None:
        self._skip()
        if self.position >= len(self.text):
            return None
        if self.text[self.position] == "'":
            self.position += 1
            characters: list[str] = []
            while self.position < len(self.text):
                character = self.text[self.position]
                if character == "'":
                    if self.position + 1 < len(self.text) and self.text[self.position + 1] == "'":
                        characters.append("'")
                        self.position += 2
                        continue
                    self.position += 1
                    return "".join(characters)
                characters.append(character)
                self.position += 1
            raise ValueError("unterminated quoted Newick label")
        start = self.position
        while self.position < len(self.text):
            character = self.text[self.position]
            if character.isspace() or character in ":,();[]":
                break
            self.position += 1
        return self.text[start:self.position] or None

    def _length(self) -> float | None:
        self._skip()
        if self._peek() != ":":
            return None
        self.position += 1
        self._skip()
        start = self.position
        while self.position < len(self.text):
            character = self.text[self.position]
            if character.isspace() or character in ",();[]":
                break
            self.position += 1
        token = self.text[start:self.position]
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"invalid Newick branch length {token!r}") from exc
        if not math.isfinite(value):
            raise ValueError("Newick branch length must be finite")
        return value

    def _node(self) -> int:
        self._skip()
        children: list[int] = []
        if self._peek() == "(":
            self.position += 1
            while True:
                children.append(self._node())
                token = self._peek()
                if token == ",":
                    self.position += 1
                    continue
                if token == ")":
                    self.position += 1
                    break
                raise ValueError(f"expected ',' or ')' in Newick tree at position {self.position}")
            label = self._label()
        else:
            label = self._label()
            if label is None:
                raise ValueError(f"expected Newick leaf label at position {self.position}")
        length = self._length()
        node_id = len(self.mutable)
        self.mutable.append(
            {
                "id": node_id,
                "label": label,
                "parent": None,
                "children": tuple(children),
                "branch_length": length,
            }
        )
        for child in children:
            self.mutable[child]["parent"] = node_id
        return node_id

    def parse(self, *, source: str | None = None) -> NewickTree:
        root = self._node()
        self._skip()
        if self._peek() == ";":
            self.position += 1
        self._skip()
        if self.position != len(self.text):
            raise ValueError(f"unexpected trailing Newick content at position {self.position}")
        return NewickTree(
            nodes=tuple(NewickNode(**record) for record in self.mutable),
            root=root,
            source=source,
        )


def parse_newick(text: str, *, source: str | None = None) -> NewickTree:
    if not text.strip():
        raise ValueError("Newick text is empty")
    return _NewickParser(text).parse(source=source)


def load_newick(path: str | Path, *, encoding: str = "utf-8") -> NewickTree:
    source = Path(path)
    return parse_newick(source.read_text(encoding=encoding), source=str(source))


# ---------------------------------------------------------------------------
# Frozen benchmark protocol and result-blind quartet sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    alignment_path: str
    alignment_sha256: str
    tree_path: str
    tree_sha256: str
    alignment_format: str | None = None
    study_doi: str | None = None
    dataset_doi: str | None = None
    license: str | None = None
    notes: str | None = None

    @classmethod
    def from_files(
        cls,
        dataset_id: str,
        alignment_path: str | Path,
        tree_path: str | Path,
        *,
        path_root: str | Path | None = None,
        **metadata: Any,
    ) -> "DatasetManifest":
        alignment = Path(alignment_path).resolve()
        tree = Path(tree_path).resolve()
        if path_root is None:
            stored_alignment = str(alignment)
            stored_tree = str(tree)
        else:
            root = Path(path_root).resolve()
            stored_alignment = Path(os.path.relpath(alignment, root)).as_posix()
            stored_tree = Path(os.path.relpath(tree, root)).as_posix()
        return cls(
            dataset_id=dataset_id,
            alignment_path=stored_alignment,
            alignment_sha256=sha256_file(alignment),
            tree_path=stored_tree,
            tree_sha256=sha256_file(tree),
            alignment_format=metadata.pop("alignment_format", None),
            study_doi=metadata.pop("study_doi", None),
            dataset_doi=metadata.pop("dataset_doi", None),
            license=metadata.pop("license", None),
            notes=metadata.pop("notes", None),
        )

    def resolved_alignment_path(self, base_dir: str | Path | None = None) -> Path:
        path = Path(self.alignment_path)
        if path.is_absolute():
            return path
        return (Path.cwd() if base_dir is None else Path(base_dir)) / path

    def resolved_tree_path(self, base_dir: str | Path | None = None) -> Path:
        path = Path(self.tree_path)
        if path.is_absolute():
            return path
        return (Path.cwd() if base_dir is None else Path(base_dir)) / path

    def verify(self, *, base_dir: str | Path | None = None) -> tuple[bool, tuple[str, ...]]:
        problems: list[str] = []
        alignment = self.resolved_alignment_path(base_dir)
        tree = self.resolved_tree_path(base_dir)
        if not alignment.is_file():
            problems.append("alignment_missing")
        elif sha256_file(alignment) != self.alignment_sha256:
            problems.append("alignment_hash_mismatch")
        if not tree.is_file():
            problems.append("tree_missing")
        elif sha256_file(tree) != self.tree_sha256:
            problems.append("tree_hash_mismatch")
        return not problems, tuple(problems)


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    version: str = "1.0"
    seed: int = 20260906
    max_quartets_per_dataset: int = 64
    quartets_per_edge: int = 2
    minimum_complete_sites: int = 200
    maximum_missing_fraction: float = 0.60
    methods: tuple[str, ...] = ("rank_tail", "p_distance", "logdet")
    resampling_modes: tuple[str, ...] = (
        "site",
        "circular_block",
        "partition_stratified",
    )
    bootstrap_replicates: int = 500
    circular_block_length: int = 25
    logdet_pseudocount: float = 0.5
    tie_absolute_tolerance: float = 1e-14
    tie_relative_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("protocol version must be nonempty")
        if self.max_quartets_per_dataset <= 0 or self.quartets_per_edge <= 0:
            raise ValueError("quartet sample limits must be positive")
        if self.minimum_complete_sites <= 0:
            raise ValueError("minimum_complete_sites must be positive")
        if not 0 <= self.maximum_missing_fraction < 1:
            raise ValueError("maximum_missing_fraction must be in [0, 1)")
        allowed_methods = {"rank_tail", "relative_rank_tail", "p_distance", "logdet"}
        unknown = set(self.methods) - allowed_methods
        if unknown:
            raise ValueError(f"unknown benchmark methods: {sorted(unknown)}")
        allowed_modes = {
            "site",
            "circular_block",
            "partition_stratified",
            "partition_block",
        }
        unknown_modes = set(self.resampling_modes) - allowed_modes
        if unknown_modes:
            raise ValueError(f"unknown resampling modes: {sorted(unknown_modes)}")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must be nonnegative")
        if self.circular_block_length <= 0:
            raise ValueError("circular_block_length must be positive")
        if self.logdet_pseudocount < 0 or not math.isfinite(self.logdet_pseudocount):
            raise ValueError("logdet_pseudocount must be finite and nonnegative")
        if self.tie_absolute_tolerance < 0 or self.tie_relative_tolerance < 0:
            raise ValueError("tie tolerances must be nonnegative")

    @property
    def digest(self) -> str:
        return canonical_sha256(self)


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkManifest:
    protocol: BenchmarkProtocol
    datasets: tuple[DatasetManifest, ...]
    corpus_name: str = "MathKernel phylogenetic benchmark"
    schema_version: str = "1"

    def __post_init__(self) -> None:
        identifiers = [dataset.dataset_id for dataset in self.datasets]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset identifiers must be unique")

    @property
    def digest(self) -> str:
        return canonical_sha256(self)

    def write(self, path: str | Path) -> None:
        payload = {
            "schema_version": self.schema_version,
            "corpus_name": self.corpus_name,
            "protocol": _json_ready(self.protocol),
            "protocol_sha256": self.protocol.digest,
            "datasets": _json_ready(self.datasets),
            "manifest_sha256": self.digest,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def verify_files(
        self,
        *,
        base_dir: str | Path | None = None,
    ) -> Mapping[str, tuple[bool, tuple[str, ...]]]:
        return {
            dataset.dataset_id: dataset.verify(base_dir=base_dir)
            for dataset in self.datasets
        }


@dataclass(frozen=True, slots=True)
class QuartetCase:
    dataset_id: str
    taxa: tuple[str, str, str, str]
    reference_split: tuple[int, int]
    source_edge: tuple[int, int] | None = None
    stratum: str = "edge"

    def __post_init__(self) -> None:
        if len(set(self.taxa)) != 4:
            raise ValueError("quartet case taxa must be distinct")
        if tuple(sorted(self.reference_split)) not in CANONICAL_QUARTET_SPLITS:
            raise ValueError(
                "reference split must be represented by the pair containing coordinate zero"
            )

    @property
    def case_id(self) -> str:
        token = canonical_sha256(
            (self.dataset_id, self.taxa, self.reference_split, self.stratum)
        )[:16]
        return f"{self.dataset_id}:{token}"

    @property
    def reference_name(self) -> str:
        return split_name(self.reference_split)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    dataset_id: str
    manifest_verified: bool
    manifest_problems: tuple[str, ...]
    alignment_taxa: int
    alignment_sites: int
    tree_taxa: int
    overlapping_taxa: int
    missing_tree_taxa: tuple[str, ...]
    extra_tree_taxa: tuple[str, ...]
    partitions: int
    informative_edges: int
    sampled_quartets: int
    eligible: bool
    rejection_reason: str | None


def _stable_rng(seed: int, *tokens: Any) -> np.random.Generator:
    digest = hashlib.sha256(canonical_json((seed, tokens)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))


def _canonical_edge_split(
    tree: NewickTree,
    edge: tuple[int, int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left, right = tree.edge_leaf_split(*edge)
    first, second = tuple(sorted(left)), tuple(sorted(right))
    return (first, second) if first < second else (second, first)


def sample_reference_quartets(
    tree: NewickTree,
    dataset_id: str,
    *,
    taxa: Iterable[str] | None = None,
    seed: int = 20260906,
    quartets_per_edge: int = 2,
    maximum: int = 64,
) -> tuple[QuartetCase, ...]:
    """Select quartets using only a reference tree and a frozen seed.

    Taxon coordinate order is independently shuffled.  The reference split is
    derived after that shuffle, so a method cannot benefit from the true split
    always being represented as ``01|23``.
    """

    if quartets_per_edge <= 0 or maximum <= 0:
        raise ValueError("quartet sample limits must be positive")
    selected_taxa = set(tree.leaf_labels if taxa is None else taxa)
    if not selected_taxa.issubset(set(tree.leaf_labels)):
        raise KeyError("requested sampling taxa are absent from the tree")
    cases: list[QuartetCase] = []
    seen: set[frozenset[str]] = set()
    edges = list(tree.informative_edges(selected_taxa))
    edges.sort(key=lambda edge: _canonical_edge_split(tree, edge))
    for edge in edges:
        side = tree._selected_leaves_on_side(edge[0], edge[1], selected_taxa)
        other = frozenset(selected_taxa - set(side))
        left, right = tuple(sorted(side)), tuple(sorted(other))
        if len(left) < 2 or len(right) < 2:
            continue
        rng = _stable_rng(seed, dataset_id, edge, left, right)
        chosen_for_edge = 0
        attempts = 0
        maximum_attempts = max(100, quartets_per_edge * 100)
        while chosen_for_edge < quartets_per_edge and attempts < maximum_attempts:
            attempts += 1
            left_pair = tuple(sorted(str(item) for item in rng.choice(left, size=2, replace=False)))
            right_pair = tuple(sorted(str(item) for item in rng.choice(right, size=2, replace=False)))
            key = frozenset(left_pair + right_pair)
            if key in seen:
                continue
            seen.add(key)
            taxa_order = list(left_pair + right_pair)
            rng.shuffle(taxa_order)
            taxa_tuple = tuple(taxa_order)
            left_indices = tuple(
                sorted(index for index, label in enumerate(taxa_tuple) if label in left_pair)
            )
            if 0 not in left_indices:
                left_indices = tuple(index for index in range(4) if index not in left_indices)
            cases.append(
                QuartetCase(
                    dataset_id=dataset_id,
                    taxa=taxa_tuple,  # type: ignore[arg-type]
                    reference_split=left_indices,  # type: ignore[arg-type]
                    source_edge=edge,
                    stratum="reference_edge",
                )
            )
            chosen_for_edge += 1
            if len(cases) >= maximum:
                return tuple(cases)
    return tuple(cases)


def audit_dataset(
    manifest: DatasetManifest,
    protocol: BenchmarkProtocol,
    *,
    base_dir: str | Path | None = None,
) -> tuple[DatasetAudit, SequenceAlignment | None, NewickTree | None, tuple[QuartetCase, ...]]:
    verified, problems = manifest.verify(base_dir=base_dir)
    if not verified:
        return (
            DatasetAudit(
                dataset_id=manifest.dataset_id,
                manifest_verified=False,
                manifest_problems=problems,
                alignment_taxa=0,
                alignment_sites=0,
                tree_taxa=0,
                overlapping_taxa=0,
                missing_tree_taxa=(),
                extra_tree_taxa=(),
                partitions=0,
                informative_edges=0,
                sampled_quartets=0,
                eligible=False,
                rejection_reason="manifest verification failed",
            ),
            None,
            None,
            (),
        )
    try:
        alignment = load_alignment(
            manifest.resolved_alignment_path(base_dir),
            format=manifest.alignment_format,
        )
        tree = load_newick(manifest.resolved_tree_path(base_dir))
    except (OSError, ValueError) as exc:
        return (
            DatasetAudit(
                dataset_id=manifest.dataset_id,
                manifest_verified=True,
                manifest_problems=(),
                alignment_taxa=0,
                alignment_sites=0,
                tree_taxa=0,
                overlapping_taxa=0,
                missing_tree_taxa=(),
                extra_tree_taxa=(),
                partitions=0,
                informative_edges=0,
                sampled_quartets=0,
                eligible=False,
                rejection_reason=f"parse failure: {exc}",
            ),
            None,
            None,
            (),
        )
    alignment_taxa = set(alignment.taxa)
    tree_taxa = set(tree.leaf_labels)
    overlap = alignment_taxa & tree_taxa
    cases = (
        sample_reference_quartets(
            tree,
            manifest.dataset_id,
            taxa=overlap,
            seed=protocol.seed,
            quartets_per_edge=protocol.quartets_per_edge,
            maximum=protocol.max_quartets_per_dataset,
        )
        if len(overlap) >= 4
        else ()
    )
    reason = None
    if alignment.datatype.upper() not in {"DNA", "NUCLEOTIDE", "STANDARD"}:
        reason = f"unsupported datatype {alignment.datatype}"
    elif len(overlap) < 4:
        reason = "fewer than four shared taxa"
    elif not cases:
        reason = "reference tree has no resolved quartet among shared taxa"
    audit = DatasetAudit(
        dataset_id=manifest.dataset_id,
        manifest_verified=True,
        manifest_problems=(),
        alignment_taxa=alignment.n_taxa,
        alignment_sites=alignment.n_sites,
        tree_taxa=len(tree_taxa),
        overlapping_taxa=len(overlap),
        missing_tree_taxa=tuple(sorted(alignment_taxa - tree_taxa)),
        extra_tree_taxa=tuple(sorted(tree_taxa - alignment_taxa)),
        partitions=len(alignment.partitions),
        informative_edges=len(tree.informative_edges(overlap)) if len(overlap) >= 4 else 0,
        sampled_quartets=len(cases),
        eligible=reason is None,
        rejection_reason=reason,
    )
    return audit, alignment, tree, tuple(cases)


# ---------------------------------------------------------------------------
# Quartet scores and tie-safe evaluation
# ---------------------------------------------------------------------------


def split_name(split: Sequence[int]) -> str:
    pair = tuple(sorted(int(index) for index in split))
    if pair not in CANONICAL_QUARTET_SPLITS:
        raise ValueError(
            "quartet split must be a two-index pair containing coordinate zero"
        )
    complement = tuple(index for index in range(4) if index not in pair)
    return f"{pair[0]}{pair[1]}|{complement[0]}{complement[1]}"


def _split_complement(split: Sequence[int]) -> tuple[int, int]:
    pair = tuple(sorted(int(index) for index in split))
    if pair not in CANONICAL_QUARTET_SPLITS:
        raise ValueError("invalid canonical quartet split")
    return tuple(index for index in range(4) if index not in pair)  # type: ignore[return-value]


def count_quartet_states(states: np.ndarray, *, alphabet_size: int = 4) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("quartet states must have shape (4, n_sites)")
    if values.shape[1] <= 0:
        raise ValueError("quartet states contain no sites")
    if np.any(values < 0) or np.any(values >= alphabet_size):
        raise ValueError("quartet state code is outside the declared alphabet")
    flat = (
        ((values[0] * alphabet_size + values[1]) * alphabet_size + values[2])
        * alphabet_size
        + values[3]
    )
    return np.bincount(flat, minlength=alphabet_size**4).reshape(
        (alphabet_size,) * 4
    ).astype(float)


def rank_tail_scores(
    states: np.ndarray,
    *,
    rank_bound: int = 4,
    relative: bool = False,
) -> Mapping[str, float]:
    tensor = count_quartet_states(states)
    probabilities = tensor / float(tensor.sum())
    results: dict[str, float] = {}
    for split in CANONICAL_QUARTET_SPLITS:
        right = _split_complement(split)
        matrix = np.transpose(probabilities, split + right).reshape(16, 16)
        results[split_name(split)] = rank_tail_frobenius(
            matrix,
            rank_bound,
            relative=relative,
        )
    return results


def pairwise_p_distance(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] <= 0:
        raise ValueError("quartet states must have shape (4, positive_n_sites)")
    distances = np.zeros((4, 4), dtype=float)
    for first in range(4):
        for second in range(first + 1, 4):
            distances[first, second] = distances[second, first] = float(
                np.mean(values[first] != values[second])
            )
    return distances


def _four_point_scores(distances: np.ndarray) -> Mapping[str, float]:
    matrix = np.asarray(distances, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("quartet distance matrix must be finite and 4x4")
    results: dict[str, float] = {}
    for split in CANONICAL_QUARTET_SPLITS:
        other = _split_complement(split)
        results[split_name(split)] = float(
            matrix[split[0], split[1]] + matrix[other[0], other[1]]
        )
    return results


def p_distance_scores(states: np.ndarray) -> Mapping[str, float]:
    return _four_point_scores(pairwise_p_distance(states))


def pairwise_logdet_distance(
    states: np.ndarray,
    *,
    pseudocount: float = 0.5,
    determinant_floor: float = 1e-300,
) -> np.ndarray:
    """Compute normalized log-det/paralinear pair distances."""

    values = np.asarray(states, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] <= 0:
        raise ValueError("quartet states must have shape (4, positive_n_sites)")
    if pseudocount < 0 or not math.isfinite(pseudocount):
        raise ValueError("logdet pseudocount must be finite and nonnegative")
    if determinant_floor <= 0 or not math.isfinite(determinant_floor):
        raise ValueError("determinant_floor must be finite and positive")
    distances = np.zeros((4, 4), dtype=float)
    for first in range(4):
        for second in range(first + 1, 4):
            table = np.full((4, 4), pseudocount, dtype=float)
            np.add.at(table, (values[first], values[second]), 1.0)
            table /= float(table.sum())
            rows = table.sum(axis=1)
            columns = table.sum(axis=0)
            denominator = math.sqrt(float(np.prod(rows) * np.prod(columns)))
            determinant = abs(float(np.linalg.det(table)))
            normalized = determinant / denominator if denominator > 0 else 0.0
            distance = -math.log(max(normalized, determinant_floor))
            distances[first, second] = distances[second, first] = distance
    return distances


def logdet_scores(
    states: np.ndarray,
    *,
    pseudocount: float = 0.5,
) -> Mapping[str, float]:
    return _four_point_scores(
        pairwise_logdet_distance(states, pseudocount=pseudocount)
    )


def score_quartet_methods(
    states: np.ndarray,
    *,
    methods: Sequence[str] = ("rank_tail", "p_distance", "logdet"),
    logdet_pseudocount: float = 0.5,
) -> Mapping[str, Mapping[str, float]]:
    result: dict[str, Mapping[str, float]] = {}
    for method in methods:
        if method == "rank_tail":
            result[method] = rank_tail_scores(states)
        elif method == "relative_rank_tail":
            result[method] = rank_tail_scores(states, relative=True)
        elif method == "p_distance":
            result[method] = p_distance_scores(states)
        elif method == "logdet":
            result[method] = logdet_scores(
                states,
                pseudocount=logdet_pseudocount,
            )
        else:
            raise ValueError(f"unsupported quartet method {method!r}")
    return result


def score_minimizers(
    scores: Mapping[str, float],
    *,
    absolute_tolerance: float = 1e-14,
    relative_tolerance: float = 1e-10,
) -> tuple[str, ...]:
    if not scores:
        raise ValueError("score mapping must be nonempty")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("score tolerances must be nonnegative")
    finite = [(name, float(value)) for name, value in scores.items() if math.isfinite(float(value))]
    if not finite:
        return tuple(sorted(scores))
    minimum = min(value for _, value in finite)
    tolerance = max(
        absolute_tolerance,
        relative_tolerance * max(1.0, abs(minimum)),
    )
    return tuple(sorted(name for name, value in finite if value <= minimum + tolerance))


def score_margin(scores: Mapping[str, float]) -> float:
    finite = sorted(float(value) for value in scores.values() if math.isfinite(float(value)))
    if len(finite) < 2:
        return float("nan")
    return finite[1] - finite[0]


# ---------------------------------------------------------------------------
# Resampling and support summaries
# ---------------------------------------------------------------------------


def site_bootstrap_indices(n_sites: int, rng: np.random.Generator) -> np.ndarray:
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    return rng.integers(0, n_sites, size=n_sites, dtype=np.int64)


def circular_block_bootstrap_indices(
    n_sites: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_sites <= 0 or block_length <= 0:
        raise ValueError("n_sites and block_length must be positive")
    blocks = math.ceil(n_sites / block_length)
    starts = rng.integers(0, n_sites, size=blocks, dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    sampled = np.concatenate([(start + offsets) % n_sites for start in starts])
    return sampled[:n_sites]


def filtered_partition_groups(
    alignment: SequenceAlignment,
    data: QuartetSiteData,
) -> Mapping[str, np.ndarray]:
    """Return disjoint complete-site strata induced by partition membership.

    NEXUS charsets can overlap.  Instead of silently double-counting sites, the
    function groups retained sites by their complete membership signature.  A
    site in both ``gene1`` and ``codon1`` therefore belongs to one stratum
    named ``gene1&codon1``.
    """

    if not alignment.partitions:
        return {"__all__": np.arange(data.n_sites, dtype=np.int64)}
    memberships: list[set[str]] = [set() for _ in range(alignment.n_sites)]
    for partition in alignment.partitions:
        for site in partition.site_indices(alignment.n_sites):
            memberships[int(site)].add(partition.name)
    groups: dict[str, list[int]] = {}
    for retained_position, original_site in enumerate(data.original_site_indices):
        names = sorted(memberships[int(original_site)])
        key = "&".join(names) if names else "__unassigned__"
        groups.setdefault(key, []).append(retained_position)
    return {
        name: np.asarray(positions, dtype=np.int64)
        for name, positions in sorted(groups.items())
        if positions
    }


def partition_stratified_bootstrap_indices(
    groups: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if not groups:
        raise ValueError("partition groups must be nonempty")
    sampled: list[np.ndarray] = []
    for name in sorted(groups):
        group = np.asarray(groups[name], dtype=np.int64)
        if group.ndim != 1:
            raise ValueError("partition groups must be one-dimensional")
        if group.size:
            sampled.append(rng.choice(group, size=group.size, replace=True))
    if not sampled:
        raise ValueError("partition groups contain no sites")
    return np.concatenate(sampled)


def partition_block_bootstrap_indices(
    groups: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample whole disjoint strata with replacement."""

    names = tuple(
        sorted(name for name, group in groups.items() if np.asarray(group).size)
    )
    if not names:
        raise ValueError("partition groups contain no sites")
    selected = rng.choice(np.asarray(names, dtype=object), size=len(names), replace=True)
    return np.concatenate(
        [np.asarray(groups[str(name)], dtype=np.int64) for name in selected]
    )


def resample_indices(
    mode: str,
    data: QuartetSiteData,
    alignment: SequenceAlignment,
    rng: np.random.Generator,
    *,
    circular_block_length: int,
) -> np.ndarray:
    if mode == "site":
        return site_bootstrap_indices(data.n_sites, rng)
    if mode == "circular_block":
        return circular_block_bootstrap_indices(
            data.n_sites,
            circular_block_length,
            rng,
        )
    groups = filtered_partition_groups(alignment, data)
    if mode == "partition_stratified":
        return partition_stratified_bootstrap_indices(groups, rng)
    if mode == "partition_block":
        return partition_block_bootstrap_indices(groups, rng)
    raise ValueError(f"unsupported resampling mode {mode!r}")


@dataclass(frozen=True, slots=True)
class BootstrapMethodSummary:
    method: str
    mode: str
    replicates: int
    unique_reference_wins: int
    reference_in_minimizer: int
    ties: int
    failures: int
    strict_support: float
    inclusive_support: float
    tie_rate: float
    failure_rate: float
    median_margin: float


@dataclass(frozen=True, slots=True)
class QuartetBenchmarkResult:
    case: QuartetCase
    complete_sites: int
    excluded_sites: int
    missing_fraction: float
    eligible: bool
    rejection_reason: str | None
    full_scores: Mapping[str, Mapping[str, float]]
    full_minimizers: Mapping[str, tuple[str, ...]]
    bootstrap_summaries: tuple[BootstrapMethodSummary, ...]
    bootstrap_records: tuple[Mapping[str, Any], ...]


def run_quartet_benchmark(
    alignment: SequenceAlignment,
    case: QuartetCase,
    protocol: BenchmarkProtocol,
    *,
    include_bootstrap_records: bool = True,
) -> QuartetBenchmarkResult:
    data = complete_nucleotide_quartet(alignment, case.taxa)
    missing_fraction = data.excluded_sites / data.total_alignment_sites
    reason: str | None = None
    if data.n_sites < protocol.minimum_complete_sites:
        reason = (
            f"only {data.n_sites} complete sites; minimum is "
            f"{protocol.minimum_complete_sites}"
        )
    elif missing_fraction > protocol.maximum_missing_fraction:
        reason = (
            f"missing fraction {missing_fraction:.6f} exceeds "
            f"{protocol.maximum_missing_fraction:.6f}"
        )
    if reason is not None:
        return QuartetBenchmarkResult(
            case=case,
            complete_sites=data.n_sites,
            excluded_sites=data.excluded_sites,
            missing_fraction=missing_fraction,
            eligible=False,
            rejection_reason=reason,
            full_scores={},
            full_minimizers={},
            bootstrap_summaries=(),
            bootstrap_records=(),
        )

    full_scores = score_quartet_methods(
        data.states,
        methods=protocol.methods,
        logdet_pseudocount=protocol.logdet_pseudocount,
    )
    full_minimizers = {
        method: score_minimizers(
            scores,
            absolute_tolerance=protocol.tie_absolute_tolerance,
            relative_tolerance=protocol.tie_relative_tolerance,
        )
        for method, scores in full_scores.items()
    }
    reference = case.reference_name
    records: list[Mapping[str, Any]] = []
    summaries: list[BootstrapMethodSummary] = []
    for mode in protocol.resampling_modes:
        counts: dict[str, dict[str, Any]] = {
            method: {
                "unique": 0,
                "inclusive": 0,
                "ties": 0,
                "failures": 0,
                "margins": [],
            }
            for method in protocol.methods
        }
        rng = _stable_rng(protocol.seed, case.case_id, mode)
        for replicate in range(protocol.bootstrap_replicates):
            indices = resample_indices(
                mode,
                data,
                alignment,
                rng,
                circular_block_length=protocol.circular_block_length,
            )
            sampled = data.states[:, indices]
            try:
                method_scores = score_quartet_methods(
                    sampled,
                    methods=protocol.methods,
                    logdet_pseudocount=protocol.logdet_pseudocount,
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                for method in protocol.methods:
                    counts[method]["failures"] += 1
                continue
            for method, scores in method_scores.items():
                minimizers = score_minimizers(
                    scores,
                    absolute_tolerance=protocol.tie_absolute_tolerance,
                    relative_tolerance=protocol.tie_relative_tolerance,
                )
                is_tie = len(minimizers) != 1
                inclusive = reference in minimizers
                unique = minimizers == (reference,)
                counts[method]["unique"] += int(unique)
                counts[method]["inclusive"] += int(inclusive)
                counts[method]["ties"] += int(is_tie)
                margin = score_margin(scores)
                counts[method]["margins"].append(margin)
                if include_bootstrap_records:
                    records.append(
                        {
                            "case_id": case.case_id,
                            "dataset_id": case.dataset_id,
                            "mode": mode,
                            "replicate": replicate,
                            "method": method,
                            "reference": reference,
                            "minimizers": minimizers,
                            "unique_reference_win": unique,
                            "reference_in_minimizer": inclusive,
                            "tie": is_tie,
                            "margin": margin,
                            **{
                                f"score_{name}": value
                                for name, value in scores.items()
                            },
                        }
                    )
        for method in protocol.methods:
            values = counts[method]
            trials = protocol.bootstrap_replicates
            margins = np.asarray(values["margins"], dtype=float)
            summaries.append(
                BootstrapMethodSummary(
                    method=method,
                    mode=mode,
                    replicates=trials,
                    unique_reference_wins=int(values["unique"]),
                    reference_in_minimizer=int(values["inclusive"]),
                    ties=int(values["ties"]),
                    failures=int(values["failures"]),
                    strict_support=float(values["unique"] / trials)
                    if trials
                    else float("nan"),
                    inclusive_support=float(values["inclusive"] / trials)
                    if trials
                    else float("nan"),
                    tie_rate=float(values["ties"] / trials)
                    if trials
                    else float("nan"),
                    failure_rate=float(values["failures"] / trials)
                    if trials
                    else float("nan"),
                    median_margin=float(np.nanmedian(margins))
                    if margins.size
                    else float("nan"),
                )
            )
    return QuartetBenchmarkResult(
        case=case,
        complete_sites=data.n_sites,
        excluded_sites=data.excluded_sites,
        missing_fraction=missing_fraction,
        eligible=True,
        rejection_reason=None,
        full_scores=full_scores,
        full_minimizers=full_minimizers,
        bootstrap_summaries=tuple(summaries),
        bootstrap_records=tuple(records),
    )


def summarize_corpus(results: Sequence[QuartetBenchmarkResult]) -> Mapping[str, Any]:
    eligible = [result for result in results if result.eligible]
    rejected = [result for result in results if not result.eligible]
    full_rows: list[dict[str, Any]] = []
    for result in eligible:
        reference = result.case.reference_name
        for method, minimizers in result.full_minimizers.items():
            full_rows.append(
                {
                    "method": method,
                    "unique_correct": minimizers == (reference,),
                    "inclusive_correct": reference in minimizers,
                    "tie": len(minimizers) != 1,
                }
            )
    method_summary: dict[str, Any] = {}
    for method in sorted({row["method"] for row in full_rows}):
        rows = [row for row in full_rows if row["method"] == method]
        method_summary[method] = {
            "cases": len(rows),
            "unique_accuracy": sum(row["unique_correct"] for row in rows) / len(rows),
            "inclusive_accuracy": sum(row["inclusive_correct"] for row in rows) / len(rows),
            "tie_rate": sum(row["tie"] for row in rows) / len(rows),
        }
    bootstrap_rows = [
        summary
        for result in eligible
        for summary in result.bootstrap_summaries
    ]
    bootstrap_summary: dict[str, Any] = {}
    for method, mode in sorted({(row.method, row.mode) for row in bootstrap_rows}):
        rows = [
            row
            for row in bootstrap_rows
            if row.method == method and row.mode == mode
        ]
        total = sum(row.replicates for row in rows)
        bootstrap_summary[f"{method}:{mode}"] = {
            "cases": len(rows),
            "replicates": total,
            "strict_support": sum(row.unique_reference_wins for row in rows) / total
            if total
            else float("nan"),
            "inclusive_support": sum(row.reference_in_minimizer for row in rows) / total
            if total
            else float("nan"),
            "tie_rate": sum(row.ties for row in rows) / total
            if total
            else float("nan"),
            "failure_rate": sum(row.failures for row in rows) / total
            if total
            else float("nan"),
        }
    return {
        "total_cases": len(results),
        "eligible_cases": len(eligible),
        "rejected_cases": len(rejected),
        "full_data": method_summary,
        "bootstrap": bootstrap_summary,
        "rejections": [
            {"case_id": result.case.case_id, "reason": result.rejection_reason}
            for result in rejected
        ],
    }


__all__ = [
    "NUCLEOTIDE_ALPHABET",
    "NUCLEOTIDE_CODE",
    "CANONICAL_QUARTET_SPLITS",
    "SiteRange",
    "AlignmentPartition",
    "SequenceAlignment",
    "QuartetSiteData",
    "NewickNode",
    "NewickTree",
    "DatasetManifest",
    "BenchmarkProtocol",
    "FrozenBenchmarkManifest",
    "QuartetCase",
    "DatasetAudit",
    "BootstrapMethodSummary",
    "QuartetBenchmarkResult",
    "sha256_bytes",
    "sha256_file",
    "canonical_json",
    "canonical_sha256",
    "parse_fasta",
    "parse_phylip",
    "parse_nexus",
    "infer_alignment_format",
    "load_alignment",
    "complete_nucleotide_quartet",
    "parse_newick",
    "load_newick",
    "sample_reference_quartets",
    "audit_dataset",
    "split_name",
    "count_quartet_states",
    "rank_tail_scores",
    "pairwise_p_distance",
    "p_distance_scores",
    "pairwise_logdet_distance",
    "logdet_scores",
    "score_quartet_methods",
    "score_minimizers",
    "score_margin",
    "site_bootstrap_indices",
    "circular_block_bootstrap_indices",
    "filtered_partition_groups",
    "partition_stratified_bootstrap_indices",
    "partition_block_bootstrap_indices",
    "resample_indices",
    "run_quartet_benchmark",
    "summarize_corpus",
]
