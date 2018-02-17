rule setup_node:
    output: "node_modules/flusight-csv-tools"
    shell: "npm install"

rule process_data:
    input: "data/external/cdc-flusight-ensemble"
    output: dir = "data/processed/cdc-flusight-ensemble",
            index = "data/processed/cdc-flusight-ensemble/index.csv"
    shell: "node ./scripts/process-ensemble-data.js {input} {output.dir}"

rule collect_actual_data:
    input: "data/processed/cdc-flusight-ensemble/index.csv"
    output: "data/processed/cdc-flusight-ensemble/actual.csv"
    shell: "node ./scripts/collect-actual-data.js {input} {output}"
