Names: Lorenzo Tomada, Gaspare Li Causi

email: ltomada@sissa.it, glicausi@sissa.it

This repository contains the final project for the course in Development Tools in Scientific Computing.


TO DO:
1) Profile runtime and memory usage, saving the results and plotting
2) Runtime vs matrix size comparison (follow in detail the instructions on the course repo)
3) Accuracy vs efficiency
4) Add missing tests

# To install using Ulysses:
```bash
source shell/submit.sh
```
The previous line will load CMake and gcc. Both are needed to compile the project.
In addition, it will enable the istallation of `mpi4py`.
After that, you can just write
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
