#!/usr/bin/env python3
import os
from argparse import ArgumentParser
import numpy as np

from core.bicrystal import Bicrystal
from core.simulation import Simulation
from utils.utils import make_dirs, get_inputs, make_crystals, compute_weights, \
                  get_xy_translation, get_xy_replications

##############################################################################

def main(infile: str, debug: bool) -> None:
    """
    Performs grand canonical optimization of GB structures.

    Args:
        infile (str): YAML file of simulation parameters.
        Defaults to params.yaml.

        debug (bool): Flag for running in DEBUG mode.
        Defaults to False.

    Returns:
        None.
    """

    # Read in parameters from YAML file.
    struct, algo = get_inputs(infile, debug)

    # Create a Simulation object to orchestrate the simulation
    sim = Simulation(struct, algo, debug)
    if debug: print(f"Starting GRIP calculations from {sim.root}")
    if debug: print(f"This process is running in {sim.cfold}")
    # Note: The logging package might make debugging print statements easier.

    # Create relevant directories like "best," "calc_proc#," etc
    make_dirs(sim.pid)

    # Use structure parameters to create upper and lower bulk slabs
    lower_0, upper_0, dlat = make_crystals(struct, debug)

    # Compute the weights for replications
    weights = compute_weights(struct)
    if debug: print(f"The weight are: {weights}")

    # Create a Bicrystal object from the two bulk slabs
    bicrystal = Bicrystal(lower_0, upper_0, struct, algo, dlat,
                          make_copy=False, debug=debug)

    ##########################################################################

    # This loop samples different GB structures
    while sim.counter < sim.nruns or not sim.nruns:

        if debug: print(f"\n~~~~~ Starting simulation iteration {sim.counter+1} ~~~~~\n")

        # Make a copy of the parent slabs for each run
        bicrystal.copy_ul()

        # Sample a random translation and shift the upper slab
        dx, dy = get_xy_translation(upper_0, rng, algo["ngrid"], sim.pid, debug)
        bicrystal.shift_upper(dx, dy)
        if debug: print(f"Translation in (x, y) = [{dx:.4f}, {dy:.4f}]")

        # Get the bounds of the GB region for MD
        bicrystal.get_bounds(algo)
        if debug: print(f"Bounds for MD simulation (lower, upper, pad): \n{bicrystal.bounds}\n")

        # Sample a replication amount and replicate bicrystal in xy directions
        rx, ry = get_xy_replications(rng, weights)
        bicrystal.replicate(rx, ry)
        if debug: print(f"Replication in (x, y) = [{rx}, {ry}]")

        # Get the number of grain boundary atoms in the upper slab
        bicrystal.get_gbplane_atoms_u()
        if debug: print(f"Num atoms per plane in upper: {bicrystal.npp_u}\n")
        if debug: print(f"Num atoms in bicrystal: {bicrystal.natoms}")

        # Create vacancies in those grain boundary atoms in the upper slab
        bicrystal.defect_upper(algo, rng)

        if debug: print(f"{bicrystal} \n")
        if debug: print(f"Num atoms in bicrystal: {bicrystal.natoms}")
        if debug: print(f"n frac = {bicrystal.n}\n")

        # Perturb the GB atoms randomly
        bicrystal.perturb_atoms(rng)
        if debug: print(f"Perturbing GB atoms by {algo['perturb_u']} and {algo['perturb_l']}\n")

        # Combine the two slabs (upper w/ defects) into a single GB structure
        bicrystal.join_gb(algo)

        # Find interstitial sites and swap atoms in GB region
        swapped_n = bicrystal.find_and_swap_inters(rng)
        if debug: print(f"Swapping {swapped_n} GB atoms with interstitial sites.\n")

        # Write the GB structure to a file that is the initial structure for LAMMPS
        input_struct_file = os.path.join("calc_procs", f"calc_proc{sim.pid+1}", "STRUC")
        bicrystal.write_gb(input_struct_file)

        if debug: print(bicrystal)
        if debug: print(f"GB structure is {bicrystal.gb}\n")

        # Sample parameters like Temperature and Numsteps for this iteration
        sim.sample_params(rng)
        if debug: print(f"The simulation parameters are T={sim.md_T}, N={sim.md_steps}")

        # Run the MD simulation to produce a final, relaxed GB structure
        sim.run_md(bicrystal, update_gb=True)

        # Get the GB energy from the last line in the final file (lammps_end_STRUC)
        sim.get_gb_energy(bicrystal)

        if debug: print(bicrystal)
        if debug: print(f"GB structure is {bicrystal.gb}\n")
        if debug: print(f"The GB energy is {bicrystal.Egb} J/m^2\n")

        # Store the energy in a list and save the file to the "best" folder
        sim.store_best_structs(bicrystal)

        if debug:# and sim.counter == 3:
            assert False, "Terminated early in DEBUG mode."

##############################################################################

if __name__ == "__main__":
    parser = ArgumentParser(description="Perform grand canonical optimization of GBs.")
    parser.add_argument("-i", "--input", type=str, default="params.yaml",
                        help="File containing structure & algorithm parameters.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in DEBUG mode, which prints variables and terminates early.")
    args = parser.parse_args()
    infile = args.input
    debug = args.debug

    if debug:
        rng = np.random.default_rng(seed=1)
    else:
        rng = np.random.default_rng()

    main(infile, debug)
