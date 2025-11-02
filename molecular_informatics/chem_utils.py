"""Cheminformatics helpers for the molecular FTIR Streamlit app."""
from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw, rdMolDescriptors

try:
    import requests
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None  # type: ignore
from urllib.parse import quote

from .ftir_data import FUNCTIONAL_GROUPS, FunctionalGroup

LOGGER = logging.getLogger(__name__)

try:
    RDLogger.DisableLog("rdApp.error")
except Exception:  # pragma: no cover - defensive; RDLogger may be absent
    pass

PUBCHEM_PROPERTY_FIELDS = "CanonicalSMILES,CID"
PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{identifier}/property/"
    f"{PUBCHEM_PROPERTY_FIELDS}/JSON"
)
PUBCHEM_PUG_VIEW_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON/"
)
REQUEST_TIMEOUT = 10  # seconds

_ACTIVITY_POSITIVE_PATTERNS = [
    re.compile(r"\bactive\b"),
    re.compile(r"\bagonist\b"),
    re.compile(r"\bantagonist\b"),
    re.compile(r"\binhibitor\b"),
    re.compile(r"\bpotent\b"),
]
_ACTIVITY_NEGATIVE_PATTERNS = [
    re.compile(r"\binactive\b"),
    re.compile(r"\bno activity\b"),
    re.compile(r"\black of activity\b"),
    re.compile(r"\bnot active\b"),
]
_SELECTIVITY_POSITIVE_PATTERNS = [
    re.compile(r"\bselective\b"),
    re.compile(r"\bselectivity\b"),
]
_SELECTIVITY_NEGATIVE_PATTERNS = [
    re.compile(r"\bnon[-\s]?selective\b"),
]
_TOXICITY_POSITIVE_PATTERNS = [
    re.compile(r"\btoxic\b"),
    re.compile(r"\btoxicity\b"),
    re.compile(r"\bhepatotoxic\b"),
    re.compile(r"\bcarcinogen(?:ic)?\b"),
    re.compile(r"\bmutagenic\b"),
    re.compile(r"\bneurotoxic\b"),
]
_TOXICITY_NEGATIVE_PATTERNS = [
    re.compile(r"\bnon[-\s]?toxic\b"),
    re.compile(r"\bnot toxic\b"),
    re.compile(r"\blow toxicity\b"),
    re.compile(r"\bminimal toxicity\b"),
]
_BIOAVAILABILITY_POSITIVE_PATTERNS = [
    re.compile(r"\bbioavailability\b"),
    re.compile(r"\bbioavailable\b"),
    re.compile(r"\bwell absorbed\b"),
    re.compile(r"\bgood absorption\b"),
    re.compile(r"\brapidly absorbed\b"),
    re.compile(r"\bhigh absorption\b"),
    re.compile(r"\befficient absorption\b"),
]
_BIOAVAILABILITY_NEGATIVE_PATTERNS = [
    re.compile(r"\blow bioavailability\b"),
    re.compile(r"\bpoor bioavailability\b"),
    re.compile(r"\blimited bioavailability\b"),
    re.compile(r"\bnot bioavailable\b"),
    re.compile(r"\bpoor absorption\b"),
    re.compile(r"\blow absorption\b"),
]


@dataclass
class MoleculeInfo:
    """Container summarising a resolved molecule."""

    smiles: str
    formula: str
    mol: Chem.Mol
    pubchem_cid: Optional[int] = None


@dataclass(frozen=True)
class PubChemMedicinalSummary:
    """Lightweight summary of medicinally relevant PubChem annotations."""

    activity: Optional[str] = None
    activity_strength: Optional[float] = None
    activity_evidence: Optional[str] = None
    selectivity: Optional[str] = None
    selectivity_strength: Optional[float] = None
    selectivity_evidence: Optional[str] = None
    toxicity: Optional[str] = None
    toxicity_strength: Optional[float] = None
    toxicity_evidence: Optional[str] = None
    bioavailability: Optional[str] = None
    bioavailability_strength: Optional[float] = None
    bioavailability_evidence: Optional[str] = None

    def has_data(self) -> bool:
        return any(
            attr is not None
            for attr in (
                self.activity,
                self.selectivity,
                self.toxicity,
                self.bioavailability,
            )
        )


class MoleculeResolutionError(RuntimeError):
    """Raised when a molecule cannot be resolved from user input."""


def _is_probable_cid(query: str) -> bool:
    """Return ``True`` if the query looks like a PubChem CID."""

    return query.isdigit()


def _fetch_pubchem_properties(identifier: str) -> Optional[Dict[str, object]]:
    """Return the first property record for a PubChem compound identifier."""

    if requests is None:
        LOGGER.warning("requests is not installed; skipping PubChem lookup")
        return None

    url = PUBCHEM_PROPERTY_URL.format(identifier=identifier)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        LOGGER.error("PubChem lookup failed for %s: %s", identifier, exc)
        return None
    except ValueError as exc:  # pragma: no cover - invalid JSON
        LOGGER.error("PubChem response decoding failed for %s: %s", identifier, exc)
        return None

    properties = data.get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return None
    return properties[0]


def _search_pubchem(query: str) -> Optional[Tuple[str, Optional[int]]]:
    """Return the canonical SMILES (and CID) for ``query`` using PubChem."""

    if _is_probable_cid(query):
        identifier = f"cid/{query}"
    else:
        identifier = f"name/{quote(query)}"

    record = _fetch_pubchem_properties(identifier)
    if not record:
        return None

    smiles = record.get("CanonicalSMILES")
    cid = record.get("CID")
    if smiles is None:
        return None
    return str(smiles), int(cid) if cid is not None else None


def _lookup_pubchem_cid_by_smiles(smiles: str) -> Optional[int]:
    """Return the PubChem CID associated with a SMILES string, if available."""

    if requests is None:
        LOGGER.warning("requests is not installed; unable to look up PubChem CID")
        return None

    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
    try:
        response = requests.post(url, data={"smiles": smiles}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        LOGGER.error("PubChem CID lookup failed for SMILES %s: %s", smiles, exc)
        return None
    except ValueError as exc:  # pragma: no cover - invalid JSON
        LOGGER.error("PubChem CID lookup decoding failed for SMILES %s: %s", smiles, exc)
        return None

    try:
        cids = data["IdentifierList"]["CID"]
    except (KeyError, TypeError):
        return None
    if not cids:
        return None
    return int(cids[0])


@lru_cache(maxsize=256)
def lookup_pubchem_cid(smiles: str) -> Optional[int]:
    """Public helper to resolve a PubChem CID from a SMILES string."""

    if requests is None:
        LOGGER.warning("requests is not installed; unable to look up PubChem CID")
        return None
    return _lookup_pubchem_cid_by_smiles(smiles)


def resolve_molecule(query: str) -> MoleculeInfo:
    """Resolve ``query`` (SMILES, name, or CID) into a :class:`MoleculeInfo`."""

    cleaned = query.strip()
    if not cleaned:
        raise MoleculeResolutionError("Input is empty.")

    smiles: Optional[str] = None
    pubchem_cid: Optional[int] = None

    # First attempt: treat the input as a SMILES string.
    mol = Chem.MolFromSmiles(cleaned)
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        if requests is not None:
            pubchem_cid = lookup_pubchem_cid(smiles)
    else:
        search_result = _search_pubchem(cleaned)
        if search_result:
            smiles, pubchem_cid = search_result
            mol = Chem.MolFromSmiles(smiles)

    if mol is None or smiles is None:
        if requests is None:
            raise MoleculeResolutionError(
                "Unable to resolve the input and PubChem lookups are disabled because"
                " the 'requests' dependency is missing."
            )
        raise MoleculeResolutionError(
            "Unable to resolve the input to a molecular structure. The text is not"
            " valid SMILES and PubChem returned no match. Check the spelling and"
            " ensure you have network access."
        )

    formula = rdMolDescriptors.CalcMolFormula(mol)
    return MoleculeInfo(smiles=smiles, formula=formula, mol=mol, pubchem_cid=pubchem_cid)


def _extract_value_strings(value: Dict) -> List[str]:
    """Normalise textual content from a PUG View value block."""

    strings: List[str] = []
    for entry in value.get("StringWithMarkup", []):
        text = entry.get("String")
        if text:
            strings.append(text)
    string_value = value.get("StringValue")
    if string_value:
        strings.append(string_value)
    numbers = value.get("Number", [])
    if isinstance(numbers, list):
        for number in numbers:
            strings.append(str(number))
    elif numbers:
        strings.append(str(numbers))
    table = value.get("Table", {})
    for row in table.get("Row", []):
        for cell in row.get("Cell", []):
            text = cell.get("String")
            if text:
                strings.append(text)
    return strings


def _extract_section_strings(section: Dict) -> List[str]:
    """Recursively collect textual content from a PUG View section."""

    strings: List[str] = []
    for info in section.get("Information", []):
        value = info.get("Value", {})
        strings.extend(_extract_value_strings(value))
    for child in section.get("Section", []):
        strings.extend(_extract_section_strings(child))
    return strings


def _collect_pug_view_text(record: Dict) -> List[str]:
    """Return all textual snippets from a PUG View record."""

    sections = record.get("Section", [])
    strings: List[str] = []
    for section in sections:
        strings.extend(_extract_section_strings(section))
    return strings


def _score_from_counts(
    positive_count: int, negative_count: int = 0, *, cap: int = 4
) -> Optional[float]:
    """Convert raw hit counts into a 0-1 intensity score."""

    net = max(positive_count - negative_count, 0)
    if net <= 0:
        return None
    return min(net / cap, 1.0)


def _collect_hits(
    snippets: Iterable[str], patterns: Iterable[re.Pattern]
) -> Tuple[int, Optional[str]]:
    """Return the number of regex hits and an example snippet."""

    count = 0
    example: Optional[str] = None
    for snippet in snippets:
        for pattern in patterns:
            matches = list(pattern.finditer(snippet))
            if matches:
                count += len(matches)
                if example is None:
                    example = textwrap.shorten(snippet.strip(), width=140, placeholder="…")
    return count, example


def _evaluate_cue(
    snippets: Iterable[str],
    positive_patterns: Iterable[re.Pattern],
    negative_patterns: Iterable[re.Pattern],
    *,
    positive_label: str,
    negative_label: Optional[str] = None,
) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Derive status, strength, and evidence for a cue from text snippets."""

    pos_count, pos_example = _collect_hits(snippets, positive_patterns)
    neg_count, neg_example = _collect_hits(snippets, negative_patterns)

    if pos_count > neg_count:
        return (
            positive_label,
            _score_from_counts(pos_count, neg_count),
            pos_example,
        )
    if negative_label and neg_count > pos_count:
        return (
            negative_label,
            _score_from_counts(neg_count, pos_count),
            neg_example,
        )
    return None, None, None


@lru_cache(maxsize=128)
def fetch_pubchem_medicinal_summary(cid: int) -> Optional[PubChemMedicinalSummary]:
    """Derive a PubChem-backed medicinal summary for a compound CID."""

    if requests is None:
        LOGGER.warning("requests is not installed; skipping PubChem medicinal lookup")
        return None

    url = PUBCHEM_PUG_VIEW_URL.format(cid=cid)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        LOGGER.error("PubChem medicinal lookup failed for CID %s: %s", cid, exc)
        return None
    except ValueError as exc:  # pragma: no cover - invalid JSON
        LOGGER.error("PubChem medicinal lookup decoding failed for CID %s: %s", cid, exc)
        return None

    record = data.get("Record", {})
    snippets = [text.lower() for text in _collect_pug_view_text(record) if text]
    if not snippets:
        return None

    activity, activity_strength, activity_evidence = _evaluate_cue(
        snippets,
        _ACTIVITY_POSITIVE_PATTERNS,
        _ACTIVITY_NEGATIVE_PATTERNS,
        positive_label="active",
        negative_label="inactive",
    )

    selectivity, selectivity_strength, selectivity_evidence = _evaluate_cue(
        snippets,
        _SELECTIVITY_POSITIVE_PATTERNS,
        _SELECTIVITY_NEGATIVE_PATTERNS,
        positive_label="selective",
    )

    toxicity, toxicity_strength, toxicity_evidence = _evaluate_cue(
        snippets,
        _TOXICITY_POSITIVE_PATTERNS,
        _TOXICITY_NEGATIVE_PATTERNS,
        positive_label="toxic",
    )

    bioavailability, bioavailability_strength, bioavailability_evidence = _evaluate_cue(
        snippets,
        _BIOAVAILABILITY_POSITIVE_PATTERNS,
        _BIOAVAILABILITY_NEGATIVE_PATTERNS,
        positive_label="bioavailable",
    )

    summary = PubChemMedicinalSummary(
        activity=activity,
        activity_strength=activity_strength,
        activity_evidence=activity_evidence,
        selectivity=selectivity,
        selectivity_strength=selectivity_strength,
        selectivity_evidence=selectivity_evidence,
        toxicity=toxicity,
        toxicity_strength=toxicity_strength,
        toxicity_evidence=toxicity_evidence,
        bioavailability=bioavailability,
        bioavailability_strength=bioavailability_strength,
        bioavailability_evidence=bioavailability_evidence,
    )
    return summary if summary.has_data() else None


def get_molecule_image(mol: Chem.Mol, size: int = 300):
    """Return a PIL image representing ``mol``."""

    return Draw.MolToImage(mol, size=(size, size))


@dataclass
class FunctionalGroupMatch:
    """Result of applying a functional group pattern to a molecule."""

    group: FunctionalGroup
    match_count: int

    @property
    def present(self) -> bool:
        return self.match_count > 0


@dataclass
class FunctionalGroupDelta:
    """Net change of a functional group across reaction participants."""

    group: FunctionalGroup
    reactant_count: int
    product_count: int

    @property
    def delta(self) -> int:
        return self.product_count - self.reactant_count


def find_functional_groups(mol: Chem.Mol) -> List[FunctionalGroupMatch]:
    """Find functional groups defined in :mod:`molecular_informatics.ftir_data`."""

    matches: List[FunctionalGroupMatch] = []
    for group in FUNCTIONAL_GROUPS:
        pattern = Chem.MolFromSmarts(group.smarts)
        if pattern is None:
            LOGGER.warning("Invalid SMARTS pattern for %s", group.name)
            continue
        hits = mol.GetSubstructMatches(pattern)
        matches.append(FunctionalGroupMatch(group=group, match_count=len(hits)))
    return matches


def aggregate_group_matches(
    match_sets: Iterable[Iterable[FunctionalGroupMatch]]
) -> List[FunctionalGroupMatch]:
    """Aggregate functional group matches across multiple molecules."""

    totals: Dict[FunctionalGroup, int] = {group: 0 for group in FUNCTIONAL_GROUPS}
    for matches in match_sets:
        for match in matches:
            totals[match.group] += match.match_count
    return [
        FunctionalGroupMatch(group=group, match_count=totals[group])
        for group in FUNCTIONAL_GROUPS
    ]


def compute_group_deltas(
    reactant_matches: Iterable[FunctionalGroupMatch],
    product_matches: Iterable[FunctionalGroupMatch],
) -> List[FunctionalGroupDelta]:
    """Compute per-functional-group changes between reactants and products."""

    reactant_lookup = {match.group: match.match_count for match in reactant_matches}
    product_lookup = {match.group: match.match_count for match in product_matches}
    deltas: List[FunctionalGroupDelta] = []
    for group in FUNCTIONAL_GROUPS:
        reactant_count = reactant_lookup.get(group, 0)
        product_count = product_lookup.get(group, 0)
        deltas.append(
            FunctionalGroupDelta(
                group=group,
                reactant_count=reactant_count,
                product_count=product_count,
            )
        )
    return deltas


def summarise_groups(matches: Iterable[FunctionalGroupMatch]) -> List[Dict[str, str]]:
    """Convert :class:`FunctionalGroupMatch` objects into display dictionaries."""

    summary = []
    for match in matches:
        group = match.group
        low, high = group.wavenumber_range
        summary.append(
            {
                "Functional group": group.name,
                "SMARTS": group.smarts,
                "Range (cm⁻¹)": f"{low}–{high}",
                "Center (cm⁻¹)": f"{group.center_wavenumber:.0f}",
                "Intensity": group.intensity,
                "Notes": group.notes,
                "Detected": "Yes" if match.present else "No",
                "Occurrences": str(match.match_count),
            }
        )
    return summary
