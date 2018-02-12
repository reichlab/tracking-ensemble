rule setup_node:
    output:
        "node_modules/flusight-csv-tools"
    shell:
        # Uses the lock file
        "npm install"
