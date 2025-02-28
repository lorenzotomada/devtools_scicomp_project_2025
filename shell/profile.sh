#!/bin/bash
python -m kernprof -l -o logs/profile_eigenvalues.dat scripts/run.py --config=experiments/config
python -m line_profiler -rmt "logs/profile_eigenvalues.dat" > logs/eigenvalues.txt
