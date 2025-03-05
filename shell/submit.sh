#!/bin/bash

module load cuda/12.1
module use cuda/12.1

pytest -v > output_pytest.txt