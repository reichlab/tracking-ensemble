rule setup_node:
    output:
        "node_modules/flusight-csv-tools"
    shell:
        # Uses the lock file
        "npm install"

rule process_data:
    input:
        "data/external/cdc-flusight-ensemble"
    output:
        "data/processed/cdc-flusight-ensemble"
    shell:
        "node ./scripts/process-ensemble-data.js {input} {output}"
