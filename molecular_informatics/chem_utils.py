"""Cheminformatics helpers for the molecular FTIR Streamlit app."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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

PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{identifier}/property/"
    "CanonicalSMILES/JSON"
)
REQUEST_TIMEOUT = 10  # seconds


@dataclass
class MoleculeInfo:
    """Container summarising a resolved molecule."""

    smiles: str
    formula: str
    mol: Chem.Mol


class MoleculeResolutionError(RuntimeError):
    """Raised when a molecule cannot be resolved from user input."""


def _is_probable_cid(query: str) -> bool:
    """Return ``True`` if the query looks like a PubChem CID."""

    return query.isdigit()


def _search_pubchem(query: str) -> Optional[str]:
    """Return the canonical SMILES for ``query`` using PubChem if available."""

    if requests is None:
        LOGGER.warning("requests is not installed; skipping PubChem lookup")
        return None

    if _is_probable_cid(query):
        identifier = f"cid/{query}"
    else:
        identifier = f"name/{quote(query)}"

    url = PUBCHEM_PROPERTY_URL.format(identifier=identifier)
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        properties = data.get("PropertyTable", {}).get("Properties", [])
        if properties:
            return properties[0].get("CanonicalSMILES")
    except requests.RequestException as exc:  # pragma: no cover - network errors
        LOGGER.error("PubChem lookup failed: %s", exc)
        return None
    except ValueError as exc:  # pragma: no cover - invalid JSON
        LOGGER.error("PubChem response decoding failed: %s", exc)
        return None

    return None


def resolve_molecule(query: str) -> MoleculeInfo:
    """Resolve ``query`` (SMILES, name, or CID) into a :class:`MoleculeInfo`."""

    cleaned = query.strip()
    if not cleaned:
        raise MoleculeResolutionError("Input is empty.")

    smiles = None

    # First attempt: treat the input as a SMILES string.
    mol = Chem.MolFromSmiles(cleaned)
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
    else:
        smiles = _search_pubchem(cleaned)
        if smiles:
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
    return MoleculeInfo(smiles=smiles, formula=formula, mol=mol)


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
