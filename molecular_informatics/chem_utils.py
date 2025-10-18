"""Cheminformatics helpers for the molecular FTIR Streamlit app."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

try:
    import pubchempy as pcp
except ImportError:  # pragma: no cover - handled gracefully at runtime
    pcp = None  # type: ignore

from .ftir_data import FUNCTIONAL_GROUPS, FunctionalGroup

LOGGER = logging.getLogger(__name__)


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

    if pcp is None:
        LOGGER.warning("pubchempy is not installed; skipping PubChem lookup")
        return None

    try:
        if _is_probable_cid(query):
            compound = pcp.Compound.from_cid(int(query))
            return compound.canonical_smiles if compound else None
        results = pcp.get_compounds(query, "name")
        return results[0].canonical_smiles if results else None
    except Exception as exc:  # pragma: no cover - network errors
        LOGGER.error("PubChem lookup failed: %s", exc)
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
        raise MoleculeResolutionError(
            "Unable to resolve the input to a molecular structure."
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
