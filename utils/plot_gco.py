import argparse
import matplotlib.pyplot as plt
from ase.io.lammpsrun import read_lammps_dump
from ase.visualize.plot import plot_atoms

from unique import clear_best, process_output

#############################################################################

def make_plot(dir_best: str = "best", hide: bool = False) -> None:
    """
    Make a GCO plot (Egb vs. n) for the best structures.

    Args:
        dir_best (str, optional): Folder to save structures in.
        Defaults to "best".

        hide (bool, optional): Whether to hide the rendered Pyplot.
        Defaults to False.

    Returns:
        None, but a Pyplot might be displayed.
    """
    file_list, ngb_list, Egb_list = process_output(dir_best)[:3]
    Egbmaxmin = max([min(Egb_list[ngb_list==n]) for n in set(ngb_list)])

    fig, ax = plt.subplots()
    ax.plot(ngb_list, Egb_list, 'o', c='k', ms=3, clip_on=True)
    ymin = ax.get_ylim()[0]
    ax.set(xlim=(0, 1),
           ylim=(ymin, Egbmaxmin + 0.2 * (Egbmaxmin - ymin)),
           xlabel=r"$[n]$", ylabel=r"$E_{\mathrm{gb}}$ (J m$^{-2}$)")

    fig.savefig(f"{dir_best}/best_grip_plot.png", dpi=300, bbox_inches="tight")
    if not hide:
        plt.show()


def view_struct(filename: str) -> None:
    """
    Use ASE and Matplotlib to visualize a GB structure.

    Args:
        filename (str): Name of the LAMMPS dump file.
    """
    s = read_lammps_dump(filename)
    fig, ax = plt.subplots()
    plot_atoms(s, ax, radii=1.0, rotation=("90x,0y,0z"))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot GCO plot (Egb vs. n) for GRIP results.")
    parser.add_argument("-f", "--folder", type=str, default="best",
                        help="Folder where GB structs are stored.")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Do not show the plot.")
    parser.add_argument("-e", "--extra", action="store_true",
                        help="Remove additional higher energy structs.")
    parser.add_argument("--file", type=str, default="",
                        help="View a structure instead of the GCO plot.")
    args = parser.parse_args()
    dir_best = args.folder
    hide = args.silent
    extra = args.extra
    filename = args.file

    clear_best(dir_best, extra, save=True)

    if filename:
        view_struct(filename)
    else:
        make_plot(dir_best, hide)
