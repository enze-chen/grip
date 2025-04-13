import os
from subprocess import Popen, PIPE, getstatusoutput

import numpy as np
from ase.io.lammpsrun import read_lammps_dump

from core.bicrystal import Bicrystal
from utils.unique import clear_best

class Simulation():
    """
    A class for organizing LAMMPS parameters and running LAMMPS.

    Attributes:
        root (str): The base path for the simulation.

        debug (bool): Flag for running in DEBUG mode.

        pid (int): The current process ID.

        cfold (str): The subfolder for the current process.

        Emult (float): Multiplicative factor for GB energy to save.

        clear_freq (int): Number of iterations before clearing duplicates.

        counter (int): Counter for tracking when to clear duplicates.

        best_Egb (float): Best GB energy value found so far.

        lmp (str): The path to the LAMMPS executable.

        md_run (float): Probably of running finite-temperature MD.

        md_steps0 (int): Minimum number of MD steps to run from input file.

        md_steps1 (int): Maximum number of MD steps to run from input file.

        md_steps (int): Sampled number of MD steps to run (if md_run=True).

        md_var (bool): Whether to vary the number of MD steps.

        Tmin (int): Minimum temperature of the MD simulation.

        Tmax (int): Maximum temperature of the MD simulation.

        md_T (int): Sampled temperature of the MD simulation (if md_run=True).

        md_style (str): Pair style argument in LAMMPS for a specific potential.

        md_coeff (str): Pair coeff argument in LAMMPS for a specific potential.

        Ecoh (float): Bulk cohesive energy for the specific potential.

        mass (float): Mass of the species.
    """

    fname_final = "lammps_end_STRUC"   # Final filename when relaxation is done

    def __init__(self,
                 struct: dict,
                 algo: dict,
                 debug: bool=False):
        """
        Constructs a Simulation object.

        Args:
            struct (dict): Dictionary of structure parameters.

            algo (dict): Dictionary of algorithm parameters.

            debug (bool, optional): Flag for running in DEBUG mode.
            Defaults to False.

        Returns:
            Simulation object
        """
        self.root = os.getcwd()
        self.debug = debug
        if not self.debug:
            if "SLURM_PROCID" in os.environ:
                self.pid = int(os.environ["SLURM_PROCID"])
            elif "PBS_TASKNUM" in os.environ:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
                self.pid = comm.Get_rank()
            else:
                self.pid = 0
        else:
            self.pid = 0
        self.cfold = os.path.join(algo["dir_calcs"], f"{algo['dir_calcs']}_{self.pid + 1}")
        self.Emult = algo["Emult"]   # for saving later
        self.counter = 0
        self.best_Egb = 1000
        self.nruns = algo["nruns"]
        self.clear_freq = algo["clear_freq"]

        self.lmp = algo["lammps_bin"]
        self.md_run = algo["MD_run"]
        if self.debug:
            self.md_steps0 = 1000
        else:
            self.md_steps0 = algo["MD_min"]
            self.md_steps1 = algo["MD_max"]
        self.md_steps = None
        self.md_var = algo["var_steps"]
        self.md_T = None
        self.Tmin = algo["Tmin"]
        self.Tmax = algo["Tmax"]

        # These definitions will have to change with binary systems
        self.symbol = struct["symbol"]
        self.md_style = struct["pair_style"]
        self.md_coeff = struct["pair_coeff"]
        self.Ecoh = struct["Ecoh"]
        self.mass = struct["mass"]   # keep in separate data file?


    def sample_params(self, rng: np.random.Generator) -> None:
        """
        Selects a random MD temperature and numsteps.

        Args:
            rng (np.random.Generator): Random number generator from NumPy.

        Returns:
            None
        """
        # Randomly choose a temperature in multiples of 100
        self.md_T = rng.choice(np.arange(self.Tmin, self.Tmax + 1, 100))
        if rng.random() > self.md_run:
            self.md_steps = 0
        elif self.md_var == 1 and not self.debug:
            # Linearly scale the number of MD steps
            self.md_steps = int(np.round(rng.integers(self.md_steps0, self.md_steps1, 
                                                      endpoint=True), -3))
        elif self.md_var == 2 and not self.debug:
            # Exponentially scale the number of MD steps
            C = np.log(self.md_steps1 / self.md_steps0)
            self.md_steps = int(np.round(self.md_steps0 *
                                         np.exp(C * rng.random()), -3))
        else:
            self.md_steps = self.md_steps0


    def run_md(self, system: Bicrystal, update_gb: bool=True) -> None:
        """
        Run a LAMMPS simulation in two steps: Optional high-temperature MD
        then relaxation. Writes the input structure with dummy GB energy
        if the LAMMPS executable path doesn't exist.

        Args:
            system (Bicrystal): Bicrystal object with a GB.

            update_gb (bool): Store the relaxed GB structure into Bicrystal object.

        Raises:
            Assertion: Temperature and num steps must be set first.

            Assertion: Bounds in the Bicrystal must be set first.

        Returns:
            None.
        """
        assert self.md_T is not None and self.md_steps is not None, \
            "Must set a temperature and number of steps first!"
        assert system.bounds is not None, "Compute GB boundaries using " + \
            "bicrystal.get_bounds() before running a simulation!"

        # Enter the calculation directory
        run_dir = os.path.join(self.root, self.cfold)
        os.chdir(run_dir)

        if os.path.exists(self.lmp) or getstatusoutput(self.lmp + " -h")[0] == 0:
            # First perform MD run
            if self.md_steps > 0:
                input1 = [self.lmp, "-i", "lammps.in_1", "-v", "T1", str(self.md_T),
                          "-v", "MDsteps", str(self.md_steps), "-v", "SEED", str(self.pid+1),
                          "-v", "STYLE", self.md_style, "-v", "COEFF", self.md_coeff,
                          "-v", "Lowerb", str(system.bounds[0]),
                          "-v", "Upperb", str(system.bounds[1]),
                          "-v", "MASS", str(self.mass), "-log", "log.lammps_1"]
                p1 = Popen(input1, stdout=PIPE, stderr=PIPE)
                p1.wait()
            else:
                # If no MD, copy the initial structure for part 2
                os.system("cp STRUC STRUC_temp")

            # Second perform static relaxation
            input2 = [self.lmp, "-i", "lammps.in_2",
                        "-v", "STYLE", self.md_style, "-v", "COEFF", self.md_coeff,
                        "-v", "Ecoh", str(self.Ecoh), "-v", "MASS", str(self.mass),
                        "-v", "Lowerb", str(system.bounds[0]),
                        "-v", "Upperb", str(system.bounds[1]),
                        "-v", "Pad", str(system.bounds[2]),
                        "-v", "f", self.fname_final, "-log", "log.lammps_2"]
            p2 = Popen(input2, stdout=PIPE, stderr=PIPE)
            p2.wait()
        else:
            print("Cannot find LAMMPS executable! Check path.")
            # If LAMMPS doesn't exist, just write a dummy output file
            self.write_dummy_lammps_dump(self.fname_final, system)
            with open(self.fname_final, "a") as f:
                f.write("Egb = 12.3456789\n")

        if update_gb:
            system.gb = read_lammps_dump(self.fname_final)
            type_list_init = system.gb.get_chemical_symbols()
            if len(set(type_list_init)) == 1:
                for a in system.gb:
                    a.symbol = self.symbol 
            system.relaxed = True

        # Iterate counter to keep track of when to clear duplicates
        self.counter += 1
        os.chdir(self.root)


    def get_gb_energy(self, system: Bicrystal) -> float:
        """
        Calculate the GB energy printed on the last line of the output file.

        Args:
            system (Bicrystal): Bicrystal object with a GB.

        Raises:
            Assertion: File must exist.

        Returns:
            Egb (float): GB energy in Joules per meter squared.
        """
        filename = os.path.join(self.cfold, "lammps_end_STRUC")
        assert os.path.exists(filename), "Final structure not found! " + \
            "Check log files from LAMMPS."

        with open(os.path.join(self.cfold, "lammps_end_STRUC"), "rb") as f:
            try:    # catch OSError in case of a one line file
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b'\n':
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode()

        tokens = last_line.split(' ')    #Example: "Egb = 3.419154496\n"
        assert tokens[0] == "Egb", print("No GB energy was written! 😞")
        Egb = float(tokens[2])
        system.Egb = Egb    # save energy into Bicrystal attribute
        return Egb


    def write_dummy_lammps_dump(self, filename: str, system: Bicrystal) -> None:
        """
        Write a Bicrystal structure to LAMMPS dump format (for debugging).

        Args:
            filename (str): Name of final LAMMPS dump file.

            system (Bicrystal): Bicrystal object with a GB.

        Raises:
            Assertion: GB must first be created using join_gb().

        Returns:
            None
        """
        assert system.gb is not None, "GB hasn't been created yet! " + \
            "Use the join_gb() method before writing."

        with open(filename, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{system.natoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp ss\n")
            for i in range(3):
                f.write(f"0.0 {system.gb.cell[i, i]}\n")
            f.write("ITEM: ATOMS id type x y z c_eng\n")
            s = system.gb.get_chemical_symbols()   # might need for binary
            s_unique_sorted = sorted(list(dict.fromkeys(s)))
            p = system.gb.get_positions()
            for i in range(system.natoms):
                idx = s_unique_sorted.index(s[i])
                f.write(f"{i+1} {idx+1} {p[i,0]} {p[i,1]} {p[i,2]} 0.0\n")


    def store_best_structs(self, system: Bicrystal, best_dir: str = "best") -> None:
        """
        Store the best structures from the simulations.

        Args:
            system (Bicrystal): Bicrystal object with GB.

            best_dir (str, optional): Directory to save structures to.
            Defaults to "best."

        Returns:
            None
        """
        if system.Egb < self.best_Egb * self.Emult:
            if system.Egb < self.best_Egb:
                self.best_Egb = system.Egb

            # Custom file name to differentiate results
            fname2best = f"lammps_{system.Egb:.3f}_{system.n:.3f}_" + \
                         f"{system.dxyz[0]:.2f}_{system.dxyz[1]:.2f}_" + \
                         f"{system.rxyz[0]:d}_{system.rxyz[1]:d}_" + \
                         f"{self.md_T:d}_{self.md_steps:d}"

            if os.name == 'nt':
                ccommand = "copy"
            else:
                ccommand = "cp"
            os.system(f"{ccommand} {os.path.join(self.cfold, self.fname_final)} " + \
                      f"{os.path.join(best_dir, fname2best)}")

            # Periodically remove duplicate structures
            if self.clear_freq:
                if len(os.listdir(best_dir)) > 4000:
                    print(f"Clearing highest energy from {best_dir} now\n")
                    clear_best(best_dir, extra=True, alpha=0.5)
                elif self.counter % self.clear_freq == 0:
                    print(f"Clearing {best_dir} now\n")
                    clear_best(best_dir)

