import os
import argparse
from subprocess import call
from typing import Tuple

import numpy as np
from numpy import ndarray
import pandas as pd

#############################################################################

def process_output(dir_best: str = 'best') -> Tuple[list, ndarray, ndarray]:
    """
    Process the output files and get the Egb and n values.

    Args:
        dir_best (str, optional): Folder to save structures in.
        Defaults to 'best'.

    Returns:
        The list of file names and NumPy arrays for the GB energy and n.
    """
    file_list = sorted([x for x in os.listdir(dir_best) if x.startswith('lammps_')])
    Egb_list, ngb_list, dx_list, dy_list = [], [], [], []
    rx_list, ry_list, T_list, steps_list = [], [], [], []

    print(f"Found {len(file_list)} files.")

    for f in file_list:
        features = f.split('_')
        Egb_list.append(float(features[1]))
        ngb_list.append(float(features[2]))
        dx_list.append(float(features[3]))
        dy_list.append(float(features[4]))
        rx_list.append(int(features[5]))
        ry_list.append(int(features[6]))
        T_list.append(int(features[7]))
        steps_list.append(int(features[8]))

    return file_list, np.array(ngb_list), np.array(Egb_list), \
           np.array(dx_list), np.array(dy_list), np.array(rx_list), np.array(ry_list), \
           np.array(T_list), np.array(steps_list)


def clear_best(dir_best: str = 'best', extra: bool = False, thresh: float = 0.003,
               alpha: float = 0.25, save: bool = False) -> None:
    """
    Periodically remove duplicate structures that are saved.

    Args:
        dir_best (str, optional): Folder that structures are saved in.
        Defaults to 'best'.

        extra (bool, optional): Whether to remove additional duplicates.
        Defaults to False.

        thresh (float, optional): Energy difference below which to remove duplicates.
        Defaults to 0.003 J/m^2.

        alpha (float, optional): Multiplicative factor of Emax to set thresholds.
        Defaults to 0.25.

        save (bool, optional): Whether to save structures into CSV.
        Defaults to False.

    Returns:
        None
    """
    # First filter by unique Egb and n, minimizing replications and translations
    file_list = sorted([x for x in os.listdir(dir_best) if x.startswith('lammps_')])
    print(f"Found {len(file_list)} files.")

    unique_dir = {}
    for fx in file_list:
        features = fx.split('_')
        Egb = features[1]
        ngb = features[2]
        dx = float(features[3])
        dy = float(features[4])
        rx = int(features[5])
        ry = int(features[6])

        # Create keys for filtering unique structures
        key1 = f"{Egb}_{ngb}"
        key2 = rx * ry
        key3 = dx**2 + dy**2

        if key1 not in unique_dir.keys():
            unique_dir[key1] = [key2, key3, fx]
        else:
            key2_old, key3_old = unique_dir[key1][0:2]
            if key2 < key2_old:
                unique_dir[key1] = [key2, key3_old, fx]
            elif (key2 == key2_old) and (key3 < key3_old):
                unique_dir[key1] = [key2, key3, fx]

    unique_files = []
    for k,v in unique_dir.items():
        unique_files.append(v[-1])

    # Determine which files are redundant then delete them
    files_to_delete = set(file_list) - set(unique_files)
    print(f"Removing {len(files_to_delete)} files.")
    for d in files_to_delete:
        try:
            os.remove(os.path.join(dir_best, d))
        except FileNotFoundError:
            pass

    # Put lists into a pandas DataFrame
    lists = process_output(dir_best)
    file_list, ngb_list, Egb_list = lists[:3]
    df = pd.DataFrame({'n':ngb_list, 'Egb':Egb_list, 'dx':lists[3], 'dy':lists[4],
                       'rx':lists[5], 'ry':lists[6], 'T':lists[7], 'steps':lists[8],
                       'f':file_list})

    # If specified, then further remove files with high energy
    if extra:
        # remove files with really high energy
        Emin = min(Egb_list)
        Egbmaxmin = max([min(Egb_list[ngb_list==n]) for n in set(ngb_list)])
        max_thresh = Egbmaxmin + alpha * (Egbmaxmin - Emin)
        print(f"Max threshold: {max_thresh}")
        to_delete = df[df['Egb'] > max_thresh]['f'].tolist()

        # remove those close in energy for same value of n
        groups = df.groupby('n')
        for name,group in groups:
            group.sort_values('Egb', inplace=True)
            Emin0 = min(group['Egb'])
            Emax0 = min(max_thresh, max(group['Egb']))
            min_thresh = alpha * (Emax0 - Emin0)
            Emin = Emin0
            for ind,row in group.iterrows():
                Ecurr = row['Egb']
                if Ecurr > Emin0 + min_thresh and Emin < Ecurr <= Emin + thresh:
                    to_delete.append(row['f'])
                else:
                    Emin = Ecurr

        to_delete = set(to_delete)
        print(f"Removing {len(to_delete)} files.")
        for f in to_delete:
            try:
                os.remove(os.path.join(dir_best, f))
            except FileNotFoundError:
                pass

    df.sort_values('Egb', inplace=True)
    if save:
        df.to_csv(f"{dir_best}/best_grip_data.csv", index=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up files from GRIP.")
    parser.add_argument("-f", "--folder", type=str, default="best",
                        help="Folder where GB structs are stored.")
    parser.add_argument("-e", "--extra", action="store_true",
                        help="Remove additional higher energy structs.")
    parser.add_argument("-t", "--thresh", type=float, default=0.003,
                        help="Egb difference to remove.")
    parser.add_argument("--alpha", type=float, default=0.25,
                        help="Fraction of Emax to set thresholds.")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save results to CSV.")
    args = parser.parse_args()
    dir_best = args.folder
    extra = args.extra
    thresh = args.thresh
    alpha = args.alpha
    save = args.save

    df = clear_best(dir_best, extra, thresh, alpha, save)
    print(df.head(15))
    #call(f'ls -lh {dir_best} | head -n 15', shell=True)
