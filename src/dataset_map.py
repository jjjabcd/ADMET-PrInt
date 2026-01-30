from __future__ import annotations

from typing import Dict, Literal

TaskType = Literal["regression", "classification"]

DATASET_TO_TYPE: Dict[str, TaskType] = {
    "Caco2_Wang": "regression",
    "PAMPA_NCATS": "classification",
    "HIA_Hou": "classification",
    "Pgp_Broccatelli": "classification",
    "Bioavailability_Ma": "classification",
    "Lipophilicity_AstraZeneca": "regression",
    "Solubility_AqSolDB": "regression",
    "HydrationFreeEnergy_FreeSolv": "regression",
    "BBB_Martins": "classification",
    "PPBR_AZ": "regression",
    "VDss_Lombardo": "regression",
    "CYP2C19_Veith": "classification",
    "CYP2D6_Veith": "classification",
    "CYP3A4_Veith": "classification",
    "CYP1A2_Veith": "classification",
    "CYP2C9_Veith": "classification",
    "CYP2C9_Substrate_CarbonMangels": "classification",
    "CYP2D6_Substrate_CarbonMangels": "classification",
    "CYP3A4_Substrate_CarbonMangels": "classification",
    "Half_Life_Obach": "regression",
    "Clearance_Hepatocyte_AZ": "regression",
    "Clearance_Microsome_AZ": "regression",
    "LD50_Zhu": "regression",
    "hERG": "classification",
    "herg_central": "classification",
    "hERG_inhib": "classification",
    "hERG_Karim": "classification",
    "AMES": "classification",
    "DILI": "classification",
    "Skin_Reaction": "classification",
    "Carcinogens_Lagunin": "classification",
    "Tox21": "classification",
    "NR-AR": "classification",
    "NR-AR-LBD": "classification",
    "NR-AhR": "classification",
    "NR-Aromatase": "classification",
    "NR-ER": "classification",
    "NR-ER-LBD": "classification",
    "NR-PPAR-gamma": "classification",
    "SR-ARE": "classification",
    "SR-ATAD5": "classification",
    "SR-HSE": "classification",
    "SR-MMP": "classification",
    "SR-p53": "classification",
    "ToxCast": "classification",
    "ClinTox": "classification",
    "admet_regression": "regression",
    "admet_classification": "classification",
}

def infer_task_type(target_col: str) -> TaskType:
    """
    Args:
        target_col (str): Target column name.

    Returns:
        TaskType: 'regression' or 'classification'

    Raises:
        KeyError: If target_col is not in DATASET_TO_TYPE.
    """
    if target_col not in DATASET_TO_TYPE:
        raise KeyError(
            f"target_col '{target_col}' not found in DATASET_TO_TYPE. "
            f"Please add it to udme/utils/dataset_map.py or pass --task_type explicitly."
        )
    return DATASET_TO_TYPE[target_col]