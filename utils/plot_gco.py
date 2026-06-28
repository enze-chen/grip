import os 
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ase.io.lammpsrun import read_lammps_dump
from ase.visualize.plot import plot_atoms

from unique import clear_best

#############################################################################

def make_plot(dir_best: str = "best", Emax: float = 0.0, hide: bool = False) -> None:
    """
    Make a GCO plot (Egb vs. n) for the best structures.

    Args:
        dir_best (str, optional): Folder to save structures in.
        Defaults to "best".

        Emax (float, optiona): Maximum value on y-axis.

        hide (bool, optional): Whether to hide the rendered Pyplot.
        Defaults to False.

    Returns:
        None, but a Pyplot might be displayed.
    """
    df = pd.read_csv(os.path.join(dir_best, "best_grip_data.csv"))
    Egbmaxmin = max([min(df["Egb"][df["n"]==n]) for n in set(df["n"])])

    plt.rcParams.update({"font.size":12})
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(df["n"], df["Egb"], 'o', c='k', ms=3, clip_on=True)
    ymin = ax.get_ylim()[0]
    ymax = Emax if Emax else Egbmaxmin + 0.2 * (Egbmaxmin - ymin)
    ax.set(xlim=(0, 1),
           ylim=(ymin, ymax),
           xlabel=r"$[n]$", ylabel=r"$E_{\mathrm{gb}}$ (J m$^{-2}$)")

    fig.savefig(f"{dir_best}/best_grip_plot.png", dpi=300, bbox_inches="tight")
    if not hide:
        plt.show()


def view_struct(filename: str, color: bool = False) -> None:
    """
    Use ASE and Matplotlib to visualize a GB structure.

    Args:
        filename (str): Name of the LAMMPS dump file.
    """
    s = read_lammps_dump(filename)
    zmid = np.mean([min(s.positions[:,2]), max(s.positions[:,2])])
    gb_thick = 20

    if color:
        eng = s.arrays['c_eng']
        eng2 = eng[(s.positions[:,2] > zmid - gb_thick) & (s.positions[:,2] < zmid + gb_thick)]
        norm = mpl.colors.Normalize(vmin=eng2.min(), vmax=eng2.max())
        cmap = plt.get_cmap('viridis')
        colors = cmap(norm(eng))
    else:
        colors = None

    fig, ax = plt.subplots(figsize=(7,7))
    plot_atoms(atoms=s, ax=ax, radii=1.6, rotation=("90x,0y,0z"), colors=colors)
    ax.set_ylim(zmid - gb_thick, zmid + gb_thick)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GCO plot (Egb vs. n) for GRIP results.")
    parser.add_argument("-f", "--folder", type=str, default="best",
                        help="Folder where GB structs are stored.")
    parser.add_argument("-E", "--Emax", type=float, 
                        help="Max Egb for y axis of plot.")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Do not show the plot.")
    parser.add_argument("-e", "--extra", action="store_true",
                        help="Remove additional higher energy structs.")
    parser.add_argument("--file", type=str, default="",
                        help="View a structure instead of the GCO plot.")
    parser.add_argument("--color", action="store_true",
                        help="Color the atoms based on per-atom energies.")
    args = parser.parse_args()
    dir_best = args.folder
    Emax = args.Emax
    hide = args.silent
    extra = args.extra
    filename = args.file
    color = args.color

    clear_best(dir_best, extra, save=True)

    if filename:
        view_struct(filename, color)
    else:
        make_plot(dir_best, Emax, hide)
