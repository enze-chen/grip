# GRand canonical Interface Predictor (GRIP)

_Authors: [Enze Chen](https://enze-chen.github.io/) (Stanford University) and
[Timofey Frolov](https://people.llnl.gov/frolov2) (Lawrence Livermore National Laboratory)_     
_Version: 0.2025.05.08_

An algorithm for performing grand canonical optimization (GCO) of interfacial
structure (e.g., grain boundaries) in crystalline materials.
It automates sampling of microscopic degrees of freedom and finite-temperature rearrangements.
The algorithm repeatedly samples different structures in two phases:
  1. Structure generation and manipulation is largely handled using the
  [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).
  2. Dynamic sampling at finite temperatures followed by relaxation is performed using
  [LAMMPS](https://www.lammps.org), although in principle other energy
  evaluation methods (e.g., density functional theory in [VASP](https://www.vasp.at))
  may be used.

Video tutorials for some of the capabilities and usage patterns can be found on YouTube in our 
[2025 TMS presentation](https://youtu.be/QtuUdnOl1k4) and [command line examples](https://youtu.be/5BwtWnQ-JR8). 🎥

------


## Dependencies
- [Python](https://www.python.org/) (3.11.5+)
- [PyYAML](https://pyyaml.org/) (6.0.1)
- [NumPy](https://numpy.org/) (1.24.4)
- [ASE](https://wiki.fysik.dtu.dk/ase/) (3.22.1)
- [LAMMPS](https://www.lammps.org) (**serial** binary only)
- [MPI for Python](https://mpi4py.readthedocs.io/en/stable) (3.1.4, for PBS resource managers only)

_Optional_
- [pandas](https://pandas.pydata.org/) (1.5.3)
- [Matplotlib](https://matplotlib.org/stable/index.html) (3.10.1)


## Usage

GRIP functions as a collection of scripts, there's no binary that you need to install or compile.
Assuming the above Python libraries are installed, clone the repo and make the 
appropriate modifications in `params.yaml` (see file for detailed comments), 
including the **path** to the LAMMPS binary on your system.
You will also need the **path** to the interatomic potential file, 
as the code reads it from disk to avoid making unnecessary copies.

If you wish, you can supply your own slabs for the bicrystal configuration as
POSCAR_LOWER and POSCAR_UPPER (in the [POSCAR](https://www.vasp.at/wiki/index.php/POSCAR)
file format).
Then call:
```python
python main.py
```
If you don't have LAMMPS or just want to test the script, you can run it with the `-d` flag.
In fact, it is recommended you run it in debug mode first to ensure the proper configuration.

See the `.examples` folder for a Slurm submission script for parallel execution (preferred).
Note that GRIP can use multiple cores, but only those on a single node (for now).


## File structure
- `main.py`: Script to launch everything.
- `params.yaml`: Simulation parameters; **you'll want to edit this.**
- `core`: Folder containing main classes (`Bicrystal`, `Simulation`, etc.)
- `utils`: Folder containing helper functions (`utils.py`, `unique.py`, etc.)
- `simul_files`: Folder containing simulation files (LAMMPS input files, etc.)
- `best`: GRIP creates this folder. All relaxed structures are stored here. The naming convention is:
`lammps_Egb_n_X-SHIFT_Y-SHIFT_X-REPS_Y-REPS_TEMP_STEPS`


Duplicate files are periodically deleted by calling `clear_best()` in `utils/unique.py`.
The default method cleans about 1-3% of files on average.
Use the `-e` flag for more aggressive cleaning (>50%).
Use the `-s` flag to save the processed results to CSV from a pandas DataFrame.

Results can be visualized by running `python utils/plot_gco.py` and it generates a 
GCO plot of $E_{\mathrm{gb}}$ vs. $[n]$.
The `.examples` folder has this plot for several boundaries.
By default executing this file will save both the results (CSV) and the figure (PNG) 
to the same folder as the GRIP output files.
Adding the `--file` flag with a filename will visualize the structure down the y-axis.


------


## Areas for improvement
- [x] Add parallelism for other job schedulers besides SLURM.
- [ ] Create more flexible workflow classes for Monte Carlo moves, energy minimization options, etc.
- [ ] Extend the code to be compatible with the parallel build of LAMMPS.
- [ ] Extend the code to work on cores across multiple compute nodes.
- [ ] Extend the compositional DOF to work with multi-component systems.
- [ ] Add in Bayesian optimization to narrow down simulation parameters.
- [ ] Incorporate ways to identify the GB atoms on the fly.
- [ ] Optimize the saving of files to reduce the memory footprint (sometimes > 1 GB).
- [ ] Improve the error handling if LAMMPS crashes (and resubmit?)


## Common errors

- `KeyError` - GRIP is expecting a key that does not exist in your `params.yaml` file.
  You may have an outdated parameters file, potentially from the `.examples` folder.
  Check the current `params.yaml` file on GitHub for the up-to-date list.
- `FileNotFoundError: No such file or directory: 'lammps_end_STRUC'` - Likely because
  LAMMPS failed to run correctly, so the anticipated `STRUC` files weren't generated.
  See log files for details.
- Incorrect number of atom types - By default the STRUC file will show the number of
  atom types equal to the unique species in the system. Some potentials require all 
  parameterized species to be specified even if fewer species are actually present.
  One solution on older ASE version is to add the `specorder` parameter to
  [`write_lammps_data()`](https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#ase.io.lammpsdata.write_lammps_data)
  to enforce the proper atom type assignments.


## Contributing

If you encounter any errors or have a suggestion, feel free to raise an Issue or Pull Request.
We'll try to respond as soon as possible!


## Citation
If you use GRIP in your work, we would appreciate a citation to the [original manuscript](https://www.nature.com/articles/s41467-024-51330-9):

> Enze Chen, Tae Wook Heo, Brandon C. Wood, Mark Asta, and Timofey Frolov.
"Grand canonically optimized grain boundary phases in hexagonal close-packed titanium."
_Nature Communications_, **15**, 7049, 2024.

or in BibTeX format:

```
@article{chen_2024_grip,
    author = {Chen, Enze and Heo, Tae Wook and Wood, Brandon C. and Asta, Mark and Frolov, Timofey},
    title = {Grand canonically optimized grain boundary phases in hexagonal close-packed titanium},
    year = {2024},
    journal = {Nature Communications},
    volume = {15},
    number = {1},
    pages = {7049},
    doi = {10.1038/s41467-024-51330-9},
}
```


## License
GRIP is distributed under the terms of the MIT license. 
All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-XXXXXX
