import os
import argparse
import yaml
import shutil

import numpy as np
import pandas as pd

import ase.io


def create_ref_dft(ref_file, gb_struct, vacuum):
    ngb = len(gb_struct)
    aa, bb, cc = gb_struct.cell
    s = ase.io.read(ref_file, format="vasp")

    # First replicate as needed
    ar, br, cr = s.cell
    rx = round(aa[0] / ar[0])
    ry = round(bb[1] / br[1])
    s *= (rx, ry, 1)

    # Now guess a cutoff
    z_guess = gb_struct.cell[2, 2] * 0.8
    s_guess = s.copy()
    del s_guess[[a.index for a in s_guess if a.position[2] > z_guess]]
    while len(s_guess) < ngb:
        z_guess += 0.05
        s_guess = s.copy()
        del s_guess[[a.index for a in s_guess if a.position[2] > z_guess]]

    if len(gb_dft) != len(s_guess):
        print(f"Beware, number of atoms in GB ({len(gb_dft)}) does not equal number of atoms in the reference ({len(s_guess)})!")

    s_guess.cell[2, 2] = z_guess + vacuum
    return s_guess



def create_gb_struct(gb_file, thickness, vacuum, symb):
    df = pd.read_csv(gb_file, skiprows=9, names=['id', 'type', 'x', 'y', 'z', 'c_eng'], sep=' ')
    zmax = df['z'].max()
    sub = df[(df['z'] > 10) & (df['z'] < zmax - 10)]
    zmid = np.mean(sub[sub['c_eng'] > 1.02 * sub['c_eng'].max()].z)
    
    s = ase.io.read(gb_file, format="lammps-dump-text")
    s.positions -= [0, 0, zmid - thickness/2]
    s.cell[2, 2] = thickness + vacuum
    del s[[a.index for a in s if (a.position[2] < 0) or (a.position[2] > thickness)]]
    for a in s:
        a.symbol = symb
    return s


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Convert GRIP outputs to POSCARs.")
    parser.add_argument("-f", "--folder", type=str, default="best",
                        help="Folder containing GRIP outputs.")
    parser.add_argument("-o", "--out_folder", type=str, default="best_poscars",
                        help="Folder containing POSCAR files.")
    parser.add_argument("-i", "--infile", type=str, default="params.yaml",
                        help="GRIP input parameters.")
    parser.add_argument("-s", "--scale", type=float, default=1.0,
                        help="Ratio of DFT to MD lattice constants.")
    parser.add_argument("-E", "--energy", type=float, default=1000.,
                        help="Grain boundary energy threshold to convert.")
    parser.add_argument("-N", "--natoms", type=int, default=200,
                        help="Number of atoms threshold for final POSCAR.")
    parser.add_argument("-t", "--thickness", type=float, default=30.0,
                        help="Height in Angstroms for the GB slice.")
    parser.add_argument("-v", "--vacuum", type=float, default=10.0,
                        help="Vacuum to add on top of POSCAR.")
    args = parser.parse_args()
    folder = args.folder
    outfolder = args.out_folder
    infile = args.infile
    scale = args.scale
    Ethresh = args.energy
    Nthresh = args.natoms
    t = args.thickness
    v = args.vacuum

    os.makedirs(outfolder, exist_ok=True)
    
    files = [x for x in os.listdir(folder) if x.startswith('lammps_')]
    with open(infile, 'r') as ff:
        params = yaml.safe_load(ff)

    cell_scaling = [scale, scale, 1]

    for f in files:
        tokens = f.split('_')
        Egb = float(tokens[1])

        s = create_gb_struct(os.path.join(folder, f), t, v, params['struct']['symbol'])
        N = len(s)
        s.set_cell(s.get_cell() * cell_scaling, scale_atoms=True)

        if N < Nthresh and Egb < Ethresh:
            final_file = os.path.join(outfolder, f"POSCAR_{N}_{'_'.join(tokens[1:])}")
            ase.io.write(final_file, s, format="vasp")

