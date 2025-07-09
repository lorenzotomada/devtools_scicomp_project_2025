#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python_script="$SCRIPT_DIR/../scripts/profiling_memory_and_time.py"
profiling_files="$SCRIPT_DIR/../Profiling_files/"*

echo "Cleaning the Profiling_files folder"
rm -f $profiling_files

echo "Running with $1 process(es)"
mpirun -np "$1" python "$python_script"
