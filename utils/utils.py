import os
import shutil
import time
import yaml
from typing import Tuple

import numpy as np
from ase.lattice.bravais import Lattice
from ase.lattice.hexagonal import HexagonalClosedPacked, Hexagonal
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic, Diamond, SimpleCubic
from ase.io.vasp import read_vasp, write_vasp

from utils.constants import P_SHIFT, Z_THRESH

#############################################################################

def get_inputs(input_data: str = "params.yaml", debug: bool = False) -> Tuple[dict, dict]:
    """
    Gather structure and system information from user.

    Args:
        input_data (str, optional): YAML file of simulation parameters.
        Defaults to params.yaml.

        debug (bool, optional): Flag for running in DEBUG mode.
        Defaults to False.

    Raises:
        Assertions: A few sanity checks on parameter values (e.g., T > 0)

    Returns:
        struct (dict): Dictionary with structure parameters.

        algo (dict): Dictionary with algorithm parameters.
    """
    with open(input_data, "r") as file:
        data = yaml.safe_load(file)
        if debug: print(f"Input params: {data}\n")

        # Validate inputs for algorithm parameters
        algo = data["algo"]
        assert isinstance(algo["ngrid"], int), "Invalid data type for ngrid, should be int."
        assert 0 <= algo["frac_min"] <= 1, "Invalid value for 'frac_min', should be float in [0,1]."
        assert 0 <= algo["frac_max"] <= 1, "Invalid value for 'frac_max', should be float in [0,1]."
        assert algo["frac_min"] <= algo["frac_max"], "Invalid; 'frac_min' must be <= 'frac_max'."

        assert algo["gb_gap"] >= 0, "Bicrystal slabs require non-negative separation."
        if algo["gb_thick"] < 5:
            print("Warning: GB region is a little thin, consider increasing 'gb_thick'.")
        if algo["pad"] < 3:
            print("Warning: Padding around GB is a little thin, consider increasing 'pad'.")

        assert algo["vacuum"] >= 0, "Bicrystal slabs require non-negative vacuum on top."
        assert algo["Emult"] >= 1, "Egb multiplier must be >= 1!"

        assert 0 <= algo["inter_p"] <= 1, "Invalid value for 'inter_p', should be float in [0,1]."
        assert algo["inter_n"] >= 0, "Number of interstitials must be non-negative!"
        if algo["inter_n"] > 0:
            assert algo["inter_t"] > 0, "Interstitial thickness must be positive for swapping!"

        assert 0 < algo["Tmin"] <= algo["Tmax"], "Invalid temperatures, should be 0 < Tmin <= Tmax!"
        assert algo["MD_min"] >= 0, "Number of MD steps must be >= 0!"
        assert algo["MD_max"] >= algo["MD_min"], "Max MD steps must be >= min steps!"

        # Validate inputs for structure parameters
        struct = data["struct"]
        if struct["cutoff"] < 20:
            print("Warning: User-generated slab might be too short to avoid interface effects.")
        assert struct["mass"] > 0, "Mass of species must be positive!"
        assert struct["a"] > 0, "Cell length must be positive!"
        assert struct["c"] > 0, "Cell length must be positive!"
        if struct["Ecoh"] > 0:
            print("Warning: Supplied Ecoh was positive! Double check?")

        for i in range(3):
            assert isinstance(struct["size0"][i], int) and struct["size0"][i] > 0, \
                   f"Minimum replications in dim {i} must be integer >= 1."
            assert isinstance(struct["size"][i], int) and struct["size"][i] > 0, \
                   f"Number of replications in dim {i} must be integer >= 1."
            assert struct["size0"][i] <= struct["size"][i], \
                   f"size0 in dim {i} cannot be larger than size!"

        assert struct["reps"] in [1, 2, 3, 4], \
            f"Repetitions flag {struct['reps']} not yet supported! Choose from [1, 2, 3, 4]."

    return struct, algo


def make_dirs(pid: int, dir_struct: str = "best", dir_calcs: str = "calc_procs") -> None:
    """
    Make a directory to store the best unique structures.

    Args:
        pid (int): Current process ID.

        dir_struct (str, optional): Name of folder to store best structures.
        Defaults to "best".

        dir_calcs (str, optional): Name of folder to store calculation files.
        Defaults to "calc_procs".

    Returns:
        None, but folders are created.
    """
    # Delete old calculation folders
    shutil.rmtree(dir_calcs, ignore_errors=True)
    time.sleep(1)

    # Make new folders for parallel calculations
    os.makedirs(dir_struct, exist_ok=True)
    os.makedirs(dir_calcs, exist_ok=True)
    time.sleep(1)

    shutil.copytree("simul_files",
                    os.path.join(dir_calcs, f"{dir_calcs}_{pid+1}"),
                    dirs_exist_ok=True)
    time.sleep(2)


def compute_dhkl(crystal: str, plane: list, a: float, c: float = 0) -> float:
    """
    Compute the interplanar spacing.

    Args:
        crystal (str): The base crystal structure.

        plane (list): The indices of the plane.

        a (float): The lattice constant.

        c (float): The lattice constant (only used for HCP).
        Default is 0.

    Raises:
        Exception: Structure must be {fcc, bcc, dc, hcp}.

    Returns:
        dhkl (float): Distance between hkl planes.
    """
    if crystal.lower() in ["fcc", "bcc", "dc"]:
        return a / np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    elif crystal.lower() in ["hcp"]:
        return 1 / np.sqrt(4/3 * (plane[0]**2 + plane[0]*plane[1] + plane[1]**2) / a**2 + \
                           plane[3]**2 / c**2)
    else:
        raise Exception(f"Crystal structure '{crystal}' is not yet supported.")


def make_crystals(struct: dict, debug: bool = False) -> Tuple[Lattice, Lattice, float]:
    """
    Make the two crystallites or read in from files. Assumes z-axis is GB normal.

    Args:
        struct (dict): Dictionary with structure parameters.

        debug (bool, optional): Flag for running in DEBUG mode.
        Defaults to False.

    Raises:
        Exception: Structure must be {fcc, bcc, dc, hcp}.

    Returns:
        Tuple[Lattice, Lattice, float]: The lower and upper slabs, respectively.
        Also returns the minimum normal component of a lattice vector.
    """
    init_size = (1, 1, struct["size"][2])
    if struct["user"]:
        upper_crystal = read_vasp("POSCAR_UPPER")
        lower_crystal = read_vasp("POSCAR_LOWER")
        dlat = struct["dlat"]
    else:
        if struct["crystal"] == "fcc":
            upper_crystal = FaceCenteredCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["upper_dirs"],
                size=init_size)
            lower_crystal = FaceCenteredCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["lower_dirs"],
                size=init_size)
        elif struct["crystal"] == "bcc":
            upper_crystal = BodyCenteredCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["upper_dirs"],
                size=init_size)
            lower_crystal = BodyCenteredCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["lower_dirs"],
                size=init_size)
        elif struct["crystal"] == "dc":
            upper_crystal = Diamond(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["upper_dirs"],
                size=init_size)
            lower_crystal = Diamond(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["lower_dirs"],
                size=init_size)
        elif struct["crystal"] == "sc":
            upper_crystal = SimpleCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["upper_dirs"],
                size=init_size)
            lower_crystal = SimpleCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["lower_dirs"],
                size=init_size)
        elif struct["crystal"] == "hcp":
            upper_crystal = HexagonalClosedPacked(
                symbol=struct["symbol"],
                latticeconstant=(struct["a"], struct["c"]),
                directions=struct["upper_dirs"],
                size=init_size)
            lower_crystal = HexagonalClosedPacked(
                symbol=struct["symbol"],
                latticeconstant=(struct["a"], struct["c"]),
                directions=struct["lower_dirs"],
                size=init_size)
        else:
            raise Exception(f"Crystal structure '{struct['crystal']}' is not yet supported.")

        # Shift atoms slightly to add tolerance
        upper_crystal.positions += [0, P_SHIFT, Z_THRESH]
        upper_crystal.wrap()
        lower_crystal.positions += [0, P_SHIFT, Z_THRESH]
        lower_crystal.wrap()

        # Chop off excess crystal to speed up simulations
        names = ["lower", "upper"]
        for i,c in enumerate([lower_crystal, upper_crystal]):
            if c.cell[2, 2] > struct["cutoff"] and struct["cutoff"]:
                nvec = struct[f"{names[i]}_dirs"][2]
                dspace = compute_dhkl(struct["crystal"], nvec, struct["a"], struct["c"])
                if debug: print(f"Interplanar spacing for {names[i]} is {dspace:.6f} A.")
                zmax = (struct["cutoff"] // dspace + 1) * dspace - Z_THRESH + min(c.positions[:, 2].round(6))
                del c[[a.index for a in c if a.position[2] > zmax]]
                c.cell[2, 2] = zmax

        # Calculate dlat for later calculation of Nplane
        if struct["crystal"] in ["fcc", "bcc", "sc"]:
            unique_z = sorted(list(set(lower_crystal.positions[:, 2].round(6))))
        elif struct["crystal"] == "dc":
            parentlat = FaceCenteredCubic(
                symbol=struct["symbol"],
                latticeconstant=struct["a"],
                directions=struct["lower_dirs"],
                size=init_size)
            unique_z = sorted(list(set(parentlat.positions[:, 2].round(6))))
        elif struct["crystal"] == "hcp":
            parentlat = Hexagonal(
                symbol=struct["symbol"],
                latticeconstant=(struct["a"], struct["c"]),
                directions=struct["lower_dirs"],
                size=init_size)
            unique_z = sorted(list(set(parentlat.positions[:, 2].round(6))))
        dlat = abs(unique_z[1] - unique_z[0])

        # Save the structures (POSCAR_*) to the current directory
        if struct["write"]:
            write_vasp("POSCAR_UPPER", upper_crystal, direct=True,
                  label=f'{struct["symbol"]} UPPER - {struct["upper_dirs"]}')
            write_vasp("POSCAR_LOWER", lower_crystal, direct=True,
                  label=f'{struct["symbol"]} LOWER - {struct["lower_dirs"]}')

    # Warn if slabs are too short in z direction (arbitrarily chosen)
    if upper_crystal.cell[2, 2] < 20:
        print(f"WARNING!! Upper slab has height {upper_crystal.cell[2, 2]:.2f}A, " + \
              "which is small. Results may be inaccurate.")
    if lower_crystal.cell[2, 2] < 20:
        print(f"WARNING!! Lower slab has height {upper_crystal.cell[2, 2]:.2f}A, " + \
              "which is small. Results may be inaccurate.")

    return lower_crystal, upper_crystal, dlat


def compute_unit_props(slab: Lattice) -> Tuple[int, int]:
    """
    Compute ngb for unit plane and the number of planes in one d-spacing.

    Args:
        slab (Lattice): One of the slabs in the bicrystal.

    Returns:
        npp_unit (int): Number of atoms in one plane in unit cell.

        npl_unit (int): Number of different planes in unit cell.
    """
    npp_unit = sum(slab.positions[:, 2] < min(slab.positions[:, 2]) + Z_THRESH)
    zpos = np.sort(np.unique((np.round(slab.positions[:,2], 5))))
    diff = [zpos[i+1] - zpos[i] for i in range(len(zpos)-1)]
    npl_unit = len(np.unique(np.round(diff, 2)))
    return npp_unit, npl_unit


def compute_weights(struct: dict) -> dict:
    """
    Compute the weights for sampling repetitions.

    Args:
        struct (dict): Dictionary with structure parameters.

    Returns:
        weights (dict): Dictionary of weights for sampling repetitions.
    """
    nx = np.arange(struct["size0"][0], struct["size"][0] + 1)
    ny = np.arange(struct["size0"][1], struct["size"][1] + 1)
    nnx = struct["size"][0] - struct["size0"][0] + 1
    nny = struct["size"][1] - struct["size0"][1] + 1
    if struct["reps"] == 1:   # sample exact replications
        wx = np.zeros(nnx)
        wx[-1] = 1
        wy = np.zeros(nny)
        wy[-1] = 1
    elif struct["reps"] == 2:  # uniformly sample replications
        wx = np.ones(nnx) / nnx
        wy = np.ones(nny) / nny
    elif struct["reps"] == 3:  # weighted negative exponential smaller
        wx = np.exp(-nx) / sum(np.exp(-nx))
        wy = np.exp(-ny) / sum(np.exp(-ny))
    else:      # weighted negative exponential larger
        wx = np.exp(-nx) / sum(np.exp(-nx))
        wy = np.exp(-ny) / sum(np.exp(-ny))
        wx = wx[::-1]
        wy = wy[::-1]
    weights = {'nx':nx, 'wx':wx, 'ny':ny, 'wy':wy}

    return weights


def get_xy_translation(slab: Lattice, rng: np.random.Generator, ngrid: int,
                       pid: int, debug: bool = False) -> Tuple[float, float]:
    """
    Calculate translations in x and y.

    Args:
        slab (Lattice): One of the slabs in the bicrystal.

        rng (Generator): Random number generator from NumPy.

        ngrid (int): Number of grid points in one dimension.

        pid (int): Current process ID.

        debug (bool, optional): Flag for running in DEBUG mode.
        Defaults to False.

    Returns:
        Tuple[float, float]: Translations in x and y, respectively.
    """
    if not debug:
        if "SLURM_NPROCS" in os.environ:
            nproc = int(os.environ["SLURM_NPROCS"])
        elif "PBS_NP" in os.environ:
            nproc = int(os.environ["PBS_NP"])
        else:
            nproc = 1
    else:
        nproc = 1

    # Create grid for translations
    grid = {"x": nproc * ngrid,
            "y": nproc * ngrid}

    dx = rng.uniform(0.0, slab.cell[0, 0])
    y_translation = np.linspace(0, slab.cell[1, 1], grid["y"] + 1)
    first = int(pid / nproc * grid["y"])
    last = int((pid + 1) / nproc * grid["y"])
    dy = rng.uniform(y_translation[first], y_translation[last])
    return dx, dy


def get_xy_replications(rng: np.random.Generator, weights: dict) -> Tuple[int, int]:
    """
    Make replications in x and y with weight probabilities.

    Args:
        rng (Generator): Random number generator from NumPy.

        weights (dict): Dictionary of weights for sampling repetitions.

    Returns:
        Tuple[int, int]: Replications in x and y, respectively.
    """
    rx = rng.choice(weights['nx'], p=weights['wx'])
    ry = rng.choice(weights['ny'], p=weights['wy'])
    return rx, ry

