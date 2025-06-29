#!/bin/bash

# Usage: ./setup_modules.sh [env]
# where [env] can be: Ulysses, workstation

env_arg="$1"

if [[ "$env_arg" == "Ulysses" ]]; then
    echo "Loading modules for Ulysses cluster..."
    module load cmake/3.29.1
    module load intel/2021.2
    module load openmpi3/3.1.4

elif [[ "$env_arg" == "2" || "$env_arg" == "workstation" ]]; then
    echo "Loading modules for local workstation..."
    module load intel/2022.2.1
    module load openmpi4/4.1.4

else
    echo "Usage: $0 [Ulysses|workstation]"
    exit 1
fi
