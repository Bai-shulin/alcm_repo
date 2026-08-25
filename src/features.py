"""Feature engineering for elemental-composition descriptors.

This module is a cleaned, modular version of the original ConstructFeatures
notebook. It intentionally preserves the published/original aggregation
definitions so that existing model inputs remain reproducible.

Important:
- The original notebook hard-coded ``Number = 324``. This implementation uses
  the actual number of rows, preventing shape mismatches.
- Several scientific definitions in the legacy notebook are preserved exactly
  and are documented in VALIDATION_NOTES.md for author review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from mendeleev import element
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

NPVALENCE = {'H': 0.0, 'He': 0.0, 'Li': 0.0, 'Be': 0.0, 'B': 3.0, 'C': 4.0, 'N': 3.0, 'O': 4.0, 'F': 5.0, 'Ne': 0.0, 'Na': 6.0, 'Mg': 6.0, 'Al': 3.0, 'Si': 4.0, 'P': 3.0, 'S': 4.0, 'Cl': 5.0, 'Ar': 0.0, 'K': 6.0, 'Ca': 6.0, 'Sc': 6.0, 'Ti': 4.0, 'V': 6.0, 'Cr': 6.0, 'Mn': 6.0, 'Fe': 6.0, 'Co': 6.0, 'Ni': 6.0, 'Cu': 6.0, 'Zn': 6.0, 'Ga': 3.0, 'Ge': 4.0, 'As': 3.0, 'Se': 4.0, 'Br': 5.0, 'Kr': 0.0, 'Rb': 6.0, 'Sr': 6.0, 'Y': 6.0, 'Zr': 4.0, 'Nb': 6.0, 'Mo': 6.0, 'Tc': 6.0, 'Ru': 6.0, 'Rh': 6.0, 'Pd': 6.0, 'Ag': 6.0, 'Cd': 6.0, 'In': 3.0, 'Sn': 4.0, 'Sb': 3.0, 'Te': 4.0, 'I': 5.0, 'Xe': 0.0, 'Cs': 6.0, 'Ba': 6.0, 'La': 6.0, 'Ce': 6.0, 'Pr': 6.0, 'Nd': 6.0, 'Pm': 6.0, 'Sm': 6.0, 'Eu': 6.0, 'Gd': 6.0, 'Tb': 6.0, 'Dy': 6.0, 'Ho': 6.0, 'Er': 6.0, 'Tm': 6.0, 'Yb': 6.0, 'Lu': 6.0, 'Hf': 4.0, 'Ta': 6.0, 'W': 6.0, 'Re': 6.0, 'Os': 6.0, 'Ir': 6.0, 'Pt': 6.0, 'Au': 6.0, 'Hg': 6.0, 'Tl': 3.0, 'Pb': 4.0, 'Bi': 3.0, 'Po': 4.0, 'At': 5.0, 'Rn': 0.0, 'Fr': 6.0, 'Ra': 6.0, 'Ac': 6.0, 'Th': 4.0, 'Pa': 6.0, 'U': 6.0, 'Np': 6.0, 'Pu': 6.0, 'Am': 6.0, 'Cm': 6.0, 'Bk': 6.0, 'Cf': 6.0, 'Es': 6.0, 'Fm': 6.0, 'Md': 6.0, 'No': 6.0, 'Lr': 6.0, 'Rf': 4.0, 'Db': 6.0, 'Sg': 6.0, 'Bh': 6.0, 'Hs': 6.0, 'Mt': 0.0, 'Ds': 0.0, 'Rg': 0.0, 'Cn': 0.0, 'Nh': 0.0, 'Fl': 0.0, 'Mc': 0.0, 'Lv': 0.0, 'Ts': 0.0, 'Og': 0.0}
NPUNFILLED_PRECURSOR = {'H': 0.0, 'He': 0.0, 'Li': 6.0, 'Be': 6.0, 'B': 1.0, 'C': 2.0, 'N': 3.0, 'O': 2.0, 'F': 1.0, 'Ne': 0.0, 'Na': 0.0, 'Mg': 0.0, 'Al': 1.0, 'Si': 2.0, 'P': 3.0, 'S': 2.0, 'Cl': 1.0, 'Ar': 0.0, 'K': 0.0, 'Ca': 0.0, 'Sc': 0.0, 'Ti': 0.0, 'V': 0.0, 'Cr': 0.0, 'Mn': 0.0, 'Fe': 0.0, 'Co': 0.0, 'Ni': 0.0, 'Cu': 0.0, 'Zn': 0.0, 'Ga': 1.0, 'Ge': 2.0, 'As': 3.0, 'Se': 2.0, 'Br': 1.0, 'Kr': 0.0, 'Rb': 0.0, 'Sr': 0.0, 'Y': 0.0, 'Zr': 0.0, 'Nb': 0.0, 'Mo': 0.0, 'Tc': 0.0, 'Ru': 0.0, 'Rh': 0.0, 'Pd': 0.0, 'Ag': 0.0, 'Cd': 0.0, 'In': 1.0, 'Sn': 2.0, 'Sb': 3.0, 'Te': 2.0, 'I': 1.0, 'Xe': 0.0, 'Cs': 0.0, 'Ba': 0.0, 'La': 0.0, 'Ce': 0.0, 'Pr': 0.0, 'Nd': 0.0, 'Pm': 0.0, 'Sm': 0.0, 'Eu': 0.0, 'Gd': 0.0, 'Tb': 0.0, 'Dy': 0.0, 'Ho': 0.0, 'Er': 0.0, 'Tm': 0.0, 'Yb': 0.0, 'Lu': 0.0, 'Hf': 0.0, 'Ta': 0.0, 'W': 0.0, 'Re': 0.0, 'Os': 0.0, 'Ir': 0.0, 'Pt': 0.0, 'Au': 0.0, 'Hg': 0.0, 'Tl': 1.0, 'Pb': 2.0, 'Bi': 3.0, 'Po': 2.0, 'At': 1.0, 'Rn': 0.0, 'Fr': 0.0, 'Ra': 0.0, 'Ac': 0.0, 'Th': 0.0, 'Pa': 0.0, 'U': 0.0, 'Np': 0.0, 'Pu': 0.0, 'Am': 0.0, 'Cm': 0.0, 'Bk': 0.0, 'Cf': 0.0, 'Es': 0.0, 'Fm': 0.0, 'Md': 0.0, 'No': 0.0, 'Lr': 0.0, 'Rf': 0.0, 'Db': 0.0, 'Sg': 0.0, 'Bh': 0.0, 'Hs': 0.0, 'Mt': 0.0, 'Ds': 0.0, 'Rg': 0.0, 'Cn': 0.0, 'Nh': 1.0, 'Fl': 2.0, 'Mc': 3.0, 'Lv': 2.0, 'Ts': 1.0, 'Og': 0.0}


@dataclass(frozen=True)
class DescriptorSpec:
    label: str
    getter: Callable[[str], float]


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _attr(name: str) -> Callable[[str], float]:
    return lambda symbol: _safe_float(getattr(element(symbol), name))


def _method(name: str) -> Callable[[str], float]:
    return lambda symbol: _safe_float(getattr(element(symbol), name)())


def _ionization(order: int) -> Callable[[str], float]:
    def getter(symbol: str) -> float:
        try:
            return _safe_float(element(symbol).ionenergies[order])
        except (KeyError, TypeError, AttributeError):
            return 0.0
    return getter


DESCRIPTORS = [
    DescriptorSpec("Lattice_Parameters", _attr("lattice_constant")),
    DescriptorSpec("Mendeleev_Number", _attr("mendeleev_number")),
    DescriptorSpec("Atomic_Weight", _attr("atomic_weight")),
    DescriptorSpec("Nvalence", _attr("nvalence")),
    DescriptorSpec("NPvalence", lambda s: NPVALENCE.get(s, 0.0)),
    DescriptorSpec("NP_Unfilled", lambda s: 6.0 - NPUNFILLED_PRECURSOR.get(s, 0.0)),
    DescriptorSpec("Atomic_Volume", _attr("atomic_volume")),
    DescriptorSpec("Fusion_Heat", _attr("fusion_heat")),
    DescriptorSpec("Boiling_Point", _attr("boiling_point")),
    DescriptorSpec("Thermal_Conductivity", _attr("thermal_conductivity")),
    DescriptorSpec("Specific_Heat_Capacity", _attr("specific_heat_capacity")),
    DescriptorSpec("Density", _attr("density")),
    DescriptorSpec("Electron_Affinity", _attr("electron_affinity")),
    DescriptorSpec("Dipole_Polarizability", _attr("dipole_polarizability")),
    DescriptorSpec("Covalent_Radius_Cordero", _attr("covalent_radius_cordero")),
    DescriptorSpec("First_Ionization_Energy", _ionization(1)),
    DescriptorSpec("Second_Ionization_Energy", _ionization(2)),
    # Preserves the original notebook behavior: feature 18 also used ionenergies[2].
    DescriptorSpec("Third_Ionization_Energy", _ionization(2)),
    DescriptorSpec("Electronegativity_Allen", _attr("en_allen")),
    DescriptorSpec("Electronegativity_Pauling", _attr("en_pauling")),
    DescriptorSpec("Electrons_Number", _attr("electrons")),
    DescriptorSpec("Heat_Formation", _attr("heat_of_formation")),
    DescriptorSpec("Vdw_Radius", _attr("vdw_radius")),
    DescriptorSpec("Atomic_Mass", _attr("mass")),
    DescriptorSpec("Melting_Point", _attr("melting_point")),
    DescriptorSpec("Hardness", _method("hardness")),
    DescriptorSpec("Softness", _method("softness")),
    DescriptorSpec("Effective_Nuclear_Charge", _method("zeff")),
    DescriptorSpec("Atomic_Number", _attr("atomic_number")),
    DescriptorSpec("ElectPhilicity", _attr("electrophilicity")),
]


def parse_composition(formula: str) -> dict[str, float]:
    """Parse the simple composition notation used by the project dataset."""
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"Invalid chemical formula: {formula!r}")

    normalized = re.sub(r"[_\-]", "+", formula)
    counts: dict[str, float] = {}
    for part in normalized.split("+"):
        for symbol, amount in re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", part):
            if symbol not in ELEMENTS:
                raise ValueError(f"Unknown element symbol {symbol!r} in {formula!r}")
            value = float(amount) if amount else 1.0
            counts[symbol] = counts.get(symbol, 0.0) + value
    if not counts:
        raise ValueError(f"No elements parsed from {formula!r}")
    return counts


def composition_matrix(formulas: pd.Series) -> pd.DataFrame:
    rows = [parse_composition(x) for x in formulas]
    return pd.DataFrame(rows).reindex(columns=ELEMENTS, fill_value=0.0).fillna(0.0)


def _present_mask(ratio: np.ndarray, prop: np.ndarray) -> np.ndarray:
    return (ratio != 0) & np.isfinite(prop) & (prop != 0)


def legacy_wam(ratio: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """Legacy 'WAM' definition from the original notebook.

    Note: the denominator is the sum of non-zero elemental property values,
    not the sum of composition fractions. This is preserved for reproducibility.
    """
    mask = _present_mask(ratio, prop)
    numerator = np.where(mask, ratio * prop, 0.0).sum(axis=1)
    denominator = np.where(mask, prop, 0.0).sum(axis=1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def legacy_wgm(ratio: np.ndarray, prop: np.ndarray) -> np.ndarray:
    out = np.zeros(ratio.shape[0], dtype=float)
    for i in range(ratio.shape[0]):
        mask = _present_mask(ratio[i], prop[i]) & (prop[i] > 0)
        weights = ratio[i, mask]
        values = prop[i, mask]
        if weights.size and weights.sum() != 0:
            out[i] = np.prod(values ** weights) ** (1.0 / weights.sum())
    return out


def legacy_wem(ratio: np.ndarray, prop: np.ndarray) -> np.ndarray:
    out = np.zeros(ratio.shape[0], dtype=float)
    for i in range(ratio.shape[0]):
        mask = (ratio[i] != 0) & np.isfinite(prop[i])
        weights = ratio[i, mask]
        values = prop[i, mask]
        value_sum = values.sum()
        if not weights.size or value_sum == 0:
            continue
        omega = values / value_sum
        product = omega * weights
        product_sum = product.sum()
        if product_sum == 0:
            continue
        split = product / product_sum
        split = split[split > 0]
        out[i] = -np.sum(split * np.log2(split))
    return out


def legacy_wsd(ratio: np.ndarray, prop: np.ndarray) -> np.ndarray:
    out = np.zeros(ratio.shape[0], dtype=float)
    for i in range(ratio.shape[0]):
        mask = (ratio[i] != 0) & np.isfinite(prop[i])
        weights = ratio[i, mask]
        values = prop[i, mask]
        if not weights.size or weights.sum() == 0:
            continue
        product = weights * values
        out[i] = np.sqrt(np.sum(weights * (values - product.mean()) ** 2) / weights.sum())
    return out


def build_descriptor_block(
    ratio_df: pd.DataFrame,
    temperature: pd.Series,
    spec: DescriptorSpec,
) -> pd.DataFrame:
    values = np.array([spec.getter(symbol) for symbol in ELEMENTS], dtype=float)
    prop = np.repeat(values[None, :], len(ratio_df), axis=0)
    ratio = ratio_df.to_numpy(dtype=float)

    wam = legacy_wam(ratio, prop)
    wgm = legacy_wgm(ratio, prop)
    wem = legacy_wem(ratio, prop)
    wsd = legacy_wsd(ratio, prop)

    # Keep eight columns per elemental descriptor, matching Txx01...Txx08.
    block = np.column_stack([
        wam, wgm, wem, wsd,
        temperature.to_numpy() * wam,
        temperature.to_numpy() * wgm,
        temperature.to_numpy() * wem,
        temperature.to_numpy() * wsd,
    ])
    return pd.DataFrame(block)


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"Crystal_structure", "temperature"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    ratios = composition_matrix(data["Crystal_structure"])
    temperature = pd.to_numeric(data["temperature"], errors="raise")

    blocks = [build_descriptor_block(ratios, temperature, spec) for spec in DESCRIPTORS]
    features = pd.concat(blocks, axis=1)

    if features.shape[1] != 240:
        raise RuntimeError(f"Expected 240 features, got {features.shape[1]}")

    names = [f"T{descriptor:02d}{stat:02d}" for descriptor in range(1, 31) for stat in range(1, 9)]
    features.columns = names
    return features, ratios


def run_feature_pipeline(
    input_csv: str | Path,
    output_dir: str | Path = ".",
    target_col: str = "N_type_carrier",
) -> dict[str, Path]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv, encoding="unicode_escape")
    if target_col not in data.columns:
        raise KeyError(f"Missing target column {target_col!r}")

    features, ratios = build_features(data)

    target_path = output_dir / "N_type_carrier.csv"
    ratio_path = output_dir / "Elem_Features.csv"
    raw_path = output_dir / "FVectors_data.csv"
    mms_path = output_dir / "FVectors_MMS.csv"
    sds_path = output_dir / "FVectors_SDS.csv"

    data[[target_col]].to_csv(target_path, index=False)
    ratios.to_csv(ratio_path, index=False)
    features.to_csv(raw_path, index=False)

    mm = MinMaxScaler().fit_transform(features)
    sd = StandardScaler().fit_transform(features)
    pd.DataFrame(mm, columns=features.columns).to_csv(mms_path, index=False)
    pd.DataFrame(sd, columns=features.columns).to_csv(sds_path, index=False)

    return {
        "target": target_path,
        "element_features": ratio_path,
        "raw_features": raw_path,
        "minmax_features": mms_path,
        "standardized_features": sds_path,
    }
