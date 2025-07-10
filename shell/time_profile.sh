#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python_script="$SCRIPT_DIR/../scripts/profiling_memory_and_time.py"
profiling_files="$SCRIPT_DIR/../Profiling_files/"*
CONFIG_FOLDER="$SCRIPT_DIR/../experiments"
CONFIG_FILE="$CONFIG_FOLDER/config.yaml"

echo "Please pass as input the following arguments: 1) number of processes 2) matrix size"
echo "Cleaning the Profiling_files folder"
rm -f $profiling_files

cp "${CONFIG_FILE}" "${CONFIG_FOLDER}/config.bak"
sed -i "s/^dim: .*/dim: $2/" $CONFIG_FILE
mv "${CONFIG_FOLDER}/config.bak" "${CONFIG_FILE}"

echo "Running with $1 process(es) with a matrix of size $2"
mpirun -np "$1" python "$python_script"