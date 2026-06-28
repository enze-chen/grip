import os
import argparse
from subprocess import call
from typing import Tuple

import numpy as np
from numpy import ndarray
import pandas as pd

#############################################################################

def parse_best_filename(fname: str):
    """
    Expected filename format: lammps_<Egb>_<n>_<dx>_<dy>_<rx>_<ry>_<T>_<steps>

    Returns dict of parsed values or None if it doesn't match.
    """
    if not fname.startswith("lammps_"):
        return None

    parts = fname.split("_")
    if len(parts) < 9:
        return None

    try:
        return {"Egb": float(parts[1]), "n": float(parts[2]),
                "dx": float(parts[3]), "dy": float(parts[4]), 
                "rx": int(parts[5]), "ry": int(parts[6]),
                "T": int(parts[7]), "steps": int(parts[8]),
                "f": fname}
    except ValueError:
        return None


def remove(dir_path: str, filenames):
    removed = 0
    for fn in filenames:
        try:
            os.remove(os.path.join(dir_path, fn))
            removed += 1
        except FileNotFoundError:
            pass
    return removed


def clear_best(dir_best: str = 'best', extra: bool = False, thresh: float = 0.003,
               alpha: float = 0.25, save: bool = False) -> pd.DataFrame:
    """
    Remove duplicate structures saved in dir_best.

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
    all_files = sorted(os.listdir(dir_best))
    rows = []
    for fn in all_files:
        r = parse_best_filename(fn)
        if r is not None:
            rows.append(r)

    df = pd.DataFrame(rows)
    print(f"Found {len(df)} parsable LAMMPS files.")

    if df.empty:
        return df

    df["area"] = df["rx"] * df["ry"]
    df["shift2"] = df["dx"] ** 2 + df["dy"] ** 2

    # Stage 1 removal of duplicates
    df_sorted = df.sort_values(["Egb", "n", "area", "shift2"], ascending=True)
    keep_1 = (df_sorted.groupby(["Egb", "n"], as_index=False).head(1)["f"].tolist())

    delete_1 = set(df["f"]) - set(keep_1)
    print(f"Stage 1 removing {len(delete_1)} duplicates.")
    remove(dir_best, delete_1)

    keep_1_set = set(keep_1)
    df["exists_after_1"] = df["f"].isin(keep_1_set)

    if not extra:
        out = df[df["exists_after_1"]].copy()
        out = out.sort_values("Egb")
        if save:
            out.to_csv(os.path.join(dir_best, "best_grip_data.csv"), index=False)
        return out

    # Stage 2: If specified, then further remove files with high energy
    df2 = df[df["exists_after_1"]].copy()
    
    Emin_global = df2["Egb"].min()
    per_n_min = df2.groupby("n")["Egb"].min()
    Egbmaxmin = per_n_min.max()


    max_thresh = Egbmaxmin + alpha * (Egbmaxmin - Emin_global)
    print(f"Stage 2 max threshold: {max_thresh}")

    to_delete = set(df2.loc[df2["Egb"] > max_thresh, "f"].tolist())


    for name,group in df.groupby("n"):
        group = group.sort_values('Egb').copy()
        Emin0 = float(group['Egb'].min())
        Emax0 = float(min(max_thresh, group['Egb'].max()))
        min_thresh = alpha * (Emax0 - Emin0)

        Emin = Emin0
        for ind,row in group.iterrows():
            Ecurr = float(row['Egb'])
            if Ecurr > Emin0 + min_thresh and Emin < Ecurr <= Emin + thresh:
                to_delete.add(row['f'])
            else:
                Emin = Ecurr

    print(f"Stage 2 removing {len(to_delete)} files.")
    remove(dir_best, to_delete)

    keep_2 = set(df2["f"]) - set(to_delete)
    out = df2[df2["f"].isin(keep_2)].copy()
    out = out.sort_values("Egb")
    if save:
        out.to_csv(os.path.join(dir_best, "best_grip_data.csv"), index=False)

    return out 


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




"""
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
"""
