Names: Gaspare Li Causi, Lorenzo Tomada

email: glicausi@sissa.it, ltomada@sissa.it

# Introduction
This repository contains the final project for the course in Development Tools in Scientific Computing.

The goal of this project is to implement an efficient eigenvalue solver.
This is done following an efficient strategy specialized for symmetric matrices, which is described in detail in the notebook `Documentation.ipynb` in the `docs` folder.

# General details
The implementation of the solver is done using `mpi4py`. Moreover, the package relies on a `C++` backend that is automatically compiled when running `python -m pip install .`.
A more detailed discussion on dependencies and on how to install the package is provided at the end of the `README.md` file.
## Repo structure
We implemented various `GitHub` workflows, which include unit testing, documentation generation and code formatting.

1. Unit tests are performed using `pytest`. They are run automatically after each push. There are three test files in the `test` folder, namely `test_eigensolvers.py` (using to test the implementation of the Lanczos method and the QR algorithm), `test_zero_finder.py` (used to ensure correctness of helper functions for the divide et impera algorithm), and `test_utils.py` (to test that some helper functions work as expected).
2. All the code is commented in detail in terms of docstrings and comments corresponding to the most salient lines of code. The documentation is generated automatially using `sphinx` at each push and deployed to `GitHub` pages.
3. After each push, the code is automatically formatted using the `black` formatter.

## Where to find important files
All the important files are in the `src/pyclassify folder`. In the root directory, the only interesting files are the `CMakeLists.txt` and the `setup.py`. Notice that the `setup.py` was added to the `pyproject.toml` file as it made it easier to automatically compile the library during installation and to deal with external dependencies, e.g. `Eigen`.

In the `src/pyclassify` folder, the file `utils.py` contains some helper functions, e.g. the ones need to check that a matrix is of the correct shape.
The `cxx_utils.cpp` file contains the implementation in `C++` of some functions that are needed in the divide and conquer algorithm (e.g. the implementations of deflation, QR method and secular solvers).
In addition, the `parallel_tridiag_eigen.py` contains the actual implementation of the divide and conquer method, while `eigenvalues.py` contains the implementation of the Lanczos algorithm.
The `zero_finder.py` just consists of a first implementation of some of the functions in `cxx_utils.cpp` and has not been removed since it is used in tests to ensure that the `C++` implementation is correct.

## What did we implement?
In order to solve an eigenvalue problem, we considered multiple strategies.
1. The most trivial one was to implement the power method in order to be able to compute (at least) the biggest eigenvalue. We then used `numba` to try and optimize it, but in this case just-in-time compilation was not extremely beneficial.The implementation of the power method is contained in `eigenvalues.py`.
2. Lanczos + QR: this is an approach (tailored to the case of symmetric matrices) to compute *all* the eigenvalues and eigenvectors. Notice that, also in the case of the QR method,`numba` was not very beneficial in terms of speed-up, resulting in a pretty slow methodology. For this reason, we implemented the QR method in `C++` and used `pybind11` to expose it to `Python`. All the code written in `C++` can be found in `cxx_utils.cpp`.
3. `CuPy` implementation of all of the above: we implemented all the above methodologies using `CuPy` to see whether using GPU could speed up computations. Since this was not the case, we commented all the lines of code involving `CuPy`, so that installation of the package is no longer required and we can use our code also on machines that do not have GPU.
4. The core of the project is the implementation (as well as a generalization of the simplified case in which $\rho=1$ considered in our reference) of the _divide et impera_ method for the computation of eigenvalues of a symmetric matrix. Some helpers were originally written in `Python` and then translated to `C++` for efficiency reasons: their original implementation is in `zero_finder.py` and is still present in the project for testing purposes. The translated version can be found in `cxx_utils.cpp`. Instead, the implementation of the actual method to compute the eigenvalues starting from a tridiagonal matrix is contained in `parallel_tridiag_eigen.py` and makes use of `mpi4py`. Notice that the implementation of deflation in `cxx_utils.cpp` is done using the `Eigen` library.

# Results
The results of the profiling (runtime vs matrix size, memory consumption, scalability, and so on) are discussed in detail in `Documentation.ipynb`.
All the scripts in the `scripts` folder are either used for profiling or to provide running examples.

## Important remark
The method that we implemented was tested thoroughly at all stages of development using `pytest`.
Nevertheless, the algorithm that we chose seems to lack robustness, meaning that there exist some matrices for which the results are not accurate (even though most of the times they are).
We are convinced that this issue is related to stability issues, as is fairly common in numerical linear algebra.

# How to run
We provide an example of running code in the `script` folder.
Assuming that you are in the root folder of the project, it sufficies to use
```bash
python scripts/mpi_running.py
```
The previous command will run the script using as configuration file (containing, e.g., matrix size and number of processes) the file `experiments/config.yaml`.
It is also possible to provide paths to other configuration files by passing the corresponding path through the `--config=path/to/file` command.

Notice that the script is *not* called using `mpirun`, but internally it uses `MPI`.
This is done by spawning a communicator inside the script.

In addition, in the `shell` folder, we provide a `submit.sbatch` file to run using `SLURM`, as well as a `submit.sh`.
They are used to perform memory profiling.

The `submit.sbatch` file is supposed to be used on Ulysses (or any other cluster using `SLURM`).
It is supposed to show how to send a job (in which our package is emplyed) using `SLURM`.
Notice, however, that due to Ulysse's problems with `MPI` the profiling for  
As a result, we also provide `submit.sh`, which is supposed to be run on a workstation.
It executes `mpirun -np [n_procs] python scripts/profile_memory.py`, basically doing the same as the `submit.sbatch` script, but without using `SLURM`.
Notice that it assumes that `shell/load_modules.sh` has already been executed (see the next section).
Examples:
```bash
sbatch shell/subsmit.sbatch
```
and
```bash
./shell/submit.sh
```

We also remark that the script to perform memory profiling `scripts/profile_memory.py` does not spam an `MPI` communicator, but is supposed to be called using `mpirun`. The reason for that is to provide a more extensive list of examples of how our package can be used.

Notice that it is possible that `scripts/mpi_running.py` will not run on systems using `SLURM` due to the fact that we are using a specific way to spawn an `MPI` communicator.
Nevertheless, the package still works: as done in `scripts/profile_memory.py`: it sufficies to run a file that can be used in combination with `mpirun` or `srun`.

# How to install:
If you are using Ulysses or a SISSA workstation, it is likely that you will need to load a couple of modules to be able to install the package.
The exact modules change according to the device you are currently using, but it is sufficient that you have `CMake`, `gcc` and `OpenMPI`.

To streamline the installation process, we provide the script `shell/load_modules.sh`.
This script loads the modules that are required on Ulysses/my workstation (according to the flag that is passed).
To use it, run:
```bash
source shell/load_modules.sh Ulysses # or source shell/load_modules.sh workstation
```
The previous line will allow the istallation of `mpi4py` and the automatic compilation of the `C++` source file used in the project.

Once the needed modules are loaded, you can regularly install via `pip` using the following command:
```bash
python -m pip install .
```

## Additional note
There is the possibility that you may need to write the following command:
```bash
conda install -c conda-forge gxx
```
Notice that you should first try *without* using that command, as the normal installation is expected to work.
This is likely a consequence of the fact that we are using a `setup.py` file in order to automatically compile a `C++` file when `pip install .` is executed.
