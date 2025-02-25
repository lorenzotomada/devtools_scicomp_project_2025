#!/bin/bash
echo "This file is supposed to be called from the root folder of the project, using './shell/scalability.sh'."
echo "Please ensure that you are in the root folder."

mkdir -p tmp_data experiments logs
touch experiments/scalability.yaml
touch tmp_data/timings.csv

echo "N,backend,time" > tmp_data/timings.csv
export LINE_PROFILE=1
N=100 # number of samples 

# Loop over N, which in this case is chosen to be the number of the first features considered by the classifier
for d in 100 500 1000 5000 10000 50000; do
    for backend in "numba" "numpy"; do
        echo "d: $d, backend: $backend" # for readability
        echo "k: 1" > experiments/scalability.yaml    
        echo "N: $N" >> experiments/scalability.yaml
        echo "backend: $backend" >> experiments/scalability.yaml
        echo "d: $d" >> experiments/scalability.yaml

        python -m kernprof -l -o tmp_data/profile.dat scripts/scalability.py --config=experiments/scalability

        TIME_TAKEN=$(python -m line_profiler -rmt tmp_data/profile.dat | awk -v backend="$backend" '
            /Function: _get_k_nearest_neighbors/ {found_knn=1}
            /Total time:/ {time=$3}
            found_knn && backend=="numpy" {print time; exit}
            found_knn && backend=="numba" {print time; exit}
        ')
        echo "d: $d,backend: $backend, time taken: $TIME_TAKEN" 
        echo "$d,$backend,$TIME_TAKEN" >> tmp_data/timings.csv
        echo '--------------------------------------------------'
    done
done

python scripts/plot_scalability.py
echo "Experiment completed!"
#rm -rf tmp_data
rm experiments/scalability.yaml