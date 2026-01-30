#!/bin/bash

# Regression
DATASET_TARGET_PAIRS=(
  "Caco2_Wang:Caco2_Wang"
  "Clearance_Hepatocyte_AZ:Clearance_Hepatocyte_AZ"
  "Clearance_Microsome_AZ:Clearance_Microsome_AZ"
  "Half_Life_Obach:Half_Life_Obach"
  "HydrationFreeEnergy_FreeSolv:HydrationFreeEnergy_FreeSolv"
  "LD50_Zhu:LD50_Zhu"
  "Lipophilicity_AstraZeneca:Lipophilicity_AstraZeneca"
  "PPBR_AZ:PPBR_AZ"
  "Solubility_AqSolDB:Solubility_AqSolDB"
  "VDss_Lombardo:VDss_Lombardo"
  "herg_central:hERG_at_10uM"
  "herg_central:hERG_at_1uM"
)

TASK_TYPE="auto"
MODEL_TYPE="gcnn"
BASE_PATH="./../tdc/data/processed"
SEED=1
ID_COL="id"
VERSION="v1"

# =============================
# Dataset / Target loop
# =============================
for PAIR in "${DATASET_TARGET_PAIRS[@]}"; do
  IFS=":" read -r DATASET_NAME TARGET_COL <<< "${PAIR}"

  echo "############################################################################"
  echo "[RUN] DATASET: ${DATASET_NAME} | TARGET: ${TARGET_COL}"
  echo "############################################################################"

  RUN_DIR="./results/${VERSION}/${MODEL_TYPE}/${DATASET_NAME}/${TARGET_COL}"
  mkdir -p "${RUN_DIR}"
  LOG_FILE="${RUN_DIR}/cv_run.log"

  python -m src.run_tdc_cv \
    --dataset_name "${DATASET_NAME}" \
    --target_col "${TARGET_COL}" \
    --task_type "${TASK_TYPE}" \
    --model_type "${MODEL_TYPE}" \
    --base_path "${BASE_PATH}" \
    --id_col "${ID_COL}" \
    --seed "${SEED}" \
    2>&1 | tee -a "${LOG_FILE}"

  echo -e "[DONE] Finished processing ${DATASET_NAME}\n"
done