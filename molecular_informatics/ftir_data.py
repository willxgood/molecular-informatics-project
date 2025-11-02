"""Reference FTIR data and pattern definitions for functional group detection."""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class FunctionalGroup:
    """Descriptor for an FTIR-relevant functional group."""

    name: str
    smarts: str
    wavenumber_range: Tuple[int, int]
    intensity: str
    notes: str

    @property
    def center_wavenumber(self) -> float:
        """Return the midpoint of the wavenumber range."""
        low, high = self.wavenumber_range
        return (low + high) / 2


# The list is intentionally concise but covers common undergraduate-level
# functional groups. Extend this list to improve coverage.
FUNCTIONAL_GROUPS: List[FunctionalGroup] = [
    FunctionalGroup(
        name="Alcohol O-H stretch",
        smarts="[OX2H]",
        wavenumber_range=(3200, 3550),
        intensity="strong, broad",
        notes="Typically associated with hydrogen-bonded hydroxyl groups.",
    ),
    FunctionalGroup(
        name="Phenol O-H stretch",
        smarts="c[OX2H]",
        wavenumber_range=(3500, 3650),
        intensity="medium",
        notes="Sharp O-H stretch observed for phenols.",
    ),
    FunctionalGroup(
        name="Carboxylic acid O-H stretch",
        smarts="C(=O)[OX2H]",
        wavenumber_range=(2500, 3300),
        intensity="strong, very broad",
        notes="Broad band often overlapping with C-H stretches.",
    ),
    FunctionalGroup(
        name="Alkane C-H stretch",
        smarts="[CH3,CH2,CH]",
        wavenumber_range=(2850, 2960),
        intensity="medium",
        notes="Asymmetric and symmetric stretching of sp3 C-H bonds.",
    ),
    FunctionalGroup(
        name="Alkene C-H stretch",
        smarts="[CH]=C",
        wavenumber_range=(3020, 3100),
        intensity="medium",
        notes="=C-H stretch characteristic of alkenes.",
    ),
    FunctionalGroup(
        name="Aromatic C-H stretch",
        smarts="[cH]",
        wavenumber_range=(3000, 3100),
        intensity="medium",
        notes="Aromatic ring hydrogen stretching vibrations.",
    ),
    FunctionalGroup(
        name="Alkyne C-H stretch",
        smarts="[C]#C[H]",
        wavenumber_range=(3260, 3330),
        intensity="strong",
        notes="Terminal alkyne C-H stretch.",
    ),
    FunctionalGroup(
        name="Carbonyl C=O stretch",
        smarts="C=O",
        wavenumber_range=(1650, 1750),
        intensity="strong",
        notes="Generic carbonyl stretch (aldehydes, ketones, acids, esters).",
    ),
    FunctionalGroup(
        name="Amide C=O stretch",
        smarts="C(=O)N",
        wavenumber_range=(1630, 1690),
        intensity="strong",
        notes="Amide I band from peptide backbone carbonyl stretch.",
    ),
    FunctionalGroup(
        name="Nitrile C≡N stretch",
        smarts="C#N",
        wavenumber_range=(2210, 2260),
        intensity="medium",
        notes="Sharp feature for nitrile groups.",
    ),
    FunctionalGroup(
        name="Nitro N-O stretch",
        smarts="[N+](=O)[O-]",
        wavenumber_range=(1500, 1600),
        intensity="strong",
        notes="Asymmetric N-O stretch of nitro groups.",
    ),
    FunctionalGroup(
        name="Amine N-H stretch",
        smarts="[NX3;H2,H1;!$(NC=O)]",
        wavenumber_range=(3300, 3500),
        intensity="medium",
        notes="Primary and secondary amine N-H stretches.",
    ),
    FunctionalGroup(
        name="Aliphatic C-N stretch",
        smarts="[CX4]-[NX3;!$(NC=O)]",
        wavenumber_range=(1020, 1250),
        intensity="medium",
        notes="C-N stretching band common to aliphatic amines.",
    ),
    FunctionalGroup(
        name="Ether C-O stretch",
        smarts="C-O-C",
        wavenumber_range=(1050, 1150),
        intensity="strong",
        notes="C-O stretch of ethers and alcohols.",
    ),
    FunctionalGroup(
        name="Aromatic C=C stretch",
        smarts="c:c",
        wavenumber_range=(1450, 1600),
        intensity="medium",
        notes="C=C skeletal vibrations of aromatic rings.",
    ),
    FunctionalGroup(
        name="Alkene C=C stretch",
        smarts="C=C",
        wavenumber_range=(1620, 1680),
        intensity="medium",
        notes="Stretching vibrations of alkenes.",
    ),
    FunctionalGroup(
        name="Imine C=N stretch",
        smarts="[CX3]=[NX2]",
        wavenumber_range=(1640, 1690),
        intensity="medium",
        notes="Characteristic C=N stretch observed for imines and Schiff bases.",
    ),
    FunctionalGroup(
        name="Thiol S-H stretch",
        smarts="[#16X2H]",
        wavenumber_range=(2550, 2600),
        intensity="medium",
        notes="Sharp S-H stretch typical for thiols and mercaptans.",
    ),
    FunctionalGroup(
        name="Sulfoxide S=O stretch",
        smarts="[SX3](=O)",
        wavenumber_range=(1030, 1070),
        intensity="strong",
        notes="Characteristic for sulfoxides and related sulfinyl groups.",
    ),
    FunctionalGroup(
        name="Sulfonyl S=O stretch",
        smarts="S(=O)(=O)",
        wavenumber_range=(1300, 1350),
        intensity="strong",
        notes="Asymmetric S=O stretch for sulfonyl and sulfate esters.",
    ),
    FunctionalGroup(
        name="Phosphate P=O stretch",
        smarts="P=O",
        wavenumber_range=(1150, 1250),
        intensity="strong",
        notes="Observed in phosphate esters and phosphonic acids.",
    ),
    FunctionalGroup(
        name="Alkyl C-F stretch",
        smarts="[CX4]-F",
        wavenumber_range=(1000, 1400),
        intensity="strong",
        notes="Broad region for alkyl fluoride C-F stretching.",
    ),
    FunctionalGroup(
        name="Aryl C-F stretch",
        smarts="c-F",
        wavenumber_range=(1200, 1250),
        intensity="strong",
        notes="C-F stretch associated with aryl fluorides.",
    ),
    FunctionalGroup(
        name="Alkyl C-Cl stretch",
        smarts="[CX4]-Cl",
        wavenumber_range=(600, 800),
        intensity="medium",
        notes="Fingerprint region for alkyl chlorides.",
    ),
    FunctionalGroup(
        name="Aryl C-Cl stretch",
        smarts="c-Cl",
        wavenumber_range=(700, 900),
        intensity="medium",
        notes="C-Cl stretch of aromatic chlorides.",
    ),
    FunctionalGroup(
        name="Alkyl C-Br stretch",
        smarts="[CX4]-Br",
        wavenumber_range=(500, 650),
        intensity="medium",
        notes="Typical for alkyl bromides in the low fingerprint region.",
    ),
    FunctionalGroup(
        name="Aryl C-Br stretch",
        smarts="c-Br",
        wavenumber_range=(500, 600),
        intensity="medium",
        notes="C-Br stretch of aryl bromides.",
    ),
    FunctionalGroup(
        name="Alkyl C-I stretch",
        smarts="[CX4]-I",
        wavenumber_range=(500, 600),
        intensity="medium",
        notes="Low-frequency stretch characteristic of alkyl iodides.",
    ),
    FunctionalGroup(
        name="Aryl C-I stretch",
        smarts="c-I",
        wavenumber_range=(500, 550),
        intensity="medium",
        notes="C-I stretch associated with aryl iodides.",
    ),
]
