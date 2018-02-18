DATA_DIR = {
    "external": "data/external/cdc-flusight-ensemble",
    "processed": "data/processed/cdc-flusight-ensemble"
}

rule setup_node:
    output: "node_modules/flusight-csv-tools"
    shell: "npm install"

rule process_ensemble_data:
    input: DATA_DIR["external"]
    output: dir = DATA_DIR["processed"],
            index = f"{DATA_DIR['processed']}/index.csv"
    shell: "node ./scripts/process-ensemble-data.js {input} {output.dir}"

rule patch_missing_files:
    input: dir = DATA_DIR["external"],
           replace = f"{DATA_DIR['external']}/model-forecasts/component-models/CU_EAKFC_SEIRS/EW19-2012-CU_EAKFC_SEIRS.csv",
           replacement = f"{DATA_DIR['external']}/model-forecasts/component-models/CU_EAKFC_SEIRS/EW18-2012-CU_EAKFC_SEIRS.csv"
    message: "Patching {input.replace} with {input.replacement}"
    shell: "cp {input.replacement} {input.replace}"

rule collect_actual_data:
    input: f"{DATA_DIR['processed']}/index.csv"
    output: f"{DATA_DIR['processed']}/actual.csv"
    shell: "node ./scripts/collect-actual-data.js {input} {output}"
