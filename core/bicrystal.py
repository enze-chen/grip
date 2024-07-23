from typing import Tuple
import random

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from ase import Atom
from ase.lattice.bravais import Lattice
from ase.io.lammpsdata import write_lammps_data

from core.interstitial import Interstitial
from utils.constants import Z_THRESH

class Bicrystal():
    """
    A class for constructing GBs in the bicrystal configuration.

    Attributes:
        lower(0) (Lattice): Lower slab of the bicrystal (unit cell).

        upper(0) (Lattice): Upper slab of the bicrystal (unit cell).

        struct (dict): Dictionary of structural parameters.

        dlat (float): Minimum normal component of a lattice vector.

        debug (bool): Flag for running in DEBUG mode.

        rxyz (tuple): Replications of the unit cell.

        dxyz (list): Translations of the upper slab.

        dupper (float): Maximum perturbation of upper atoms.

        dlower (float): Maximum perturbation of lower atoms.

        npp_u (int): Number of atoms in one plane in full upper cell (post replications).

        gbplane_ids_u (ndarray): Array of indices of atoms in GB (from upper).

        gbplane_pos_u (ndarray): Array of positions of atoms in GB (from upper).

        bounds (list): z coordinates of [lower, upper, pad] of GB region.

        gb (Lattice): GB structure with upper and lower slabs joined together.

        n (float): n parameter for planar fraction. 0 <= n <= 1.

        Egb (float): Grain boundary energy in Joules per meter squared.

        inter_n (int): Number of interstitial sites to swap.

        interstitials (list): List of Interstitial site objects.
    """

    def __init__(self,
                 lower: Lattice,
                 upper: Lattice,
                 struct: dict,
                 algo: dict,
                 dlat: float,
                 make_copy: bool = True,
                 debug: bool = False):
        """
        Constructs a Bicrystal object.

        Args:
            lower (Lattice): Lower slab of the bicrystal.

            upper (Lattice): Upper slab of the bicrystal.

            struct (dict): Dictionary of structure parameters.

            algo (dict): Dictionary of algorithm parameters.

            npp_unit (int): Number of atoms in one plane in unit cell.

            npl_unit (int): Number of different planes in unit cell.

            dlat (float): Minimum normal component of a lattice vector.

            make_copy (bool, optional): Make a copy of lower and upper unit cells.
            Defaults to True.

            debug (bool, optional): Flag for running in DEBUG mode.
            Defaults to False.

        Returns:
            Bicrystal object
        """
        self.lower0 = lower
        self.upper0 = upper
        self.struct = struct
        self.algo = algo
        self.dlat = dlat
        self.debug = debug

        self.rxyz = (1, 1, 1)
        self.dxyz = [0, 0, 0]
        self.dupper = algo["perturb_u"]
        self.dlower = algo["perturb_l"]
        self.npp_u = None
        self.gbplane_ids_u = None
        self.gbplane_pos_u = None
        self.bounds = None
        self.gb = None
        self.relaxed = False
        self.n = None
        self.Egb = None
        self.inter_n = 0
        self.interstitials = []

        if make_copy:
            self.copy_ul()


    def copy_ul(self) -> None:
        """ Copy the lower and upper unit slabs for subsequent calculations. """
        self.lower = self.lower0.copy()
        self.upper = self.upper0.copy()


    def __repr__(self):
        """ String representation using symbols of lower and upper slabs. """
        if self.relaxed:
            suffix = "relaxed"
        elif self.gb:
            suffix = "joined"
        else:
            suffix = "unjoined"
        return f"{self.__class__.__name__}({self.lower.symbols}, {self.upper.symbols}) {suffix}"


    def __eq__(self, other):
        """ Two Bicrystals are equal if their upper and lower slabs are equal. """
        try:
            return self.upper == other.upper and self.lower == other.lower
        except:
            return self.upper0 == other.upper0 and self.lower0 == other.lower0


    def __bool__(self):
        """ Evaluates to True if the slabs have been joined. """
        return bool(self.gb)


    def __len__(self):
        """ The length of a Bicrystal is just the number of atoms. """
        return self.lower.__len__() + self.upper.__len__() + self.inter_n


    @property
    def natoms(self) -> int:
        """ Return the number of atoms in the Bicrystal. """
        return self.__len__()

    @property
    def z(self) -> float:
        return self.gb.cell.lengths()[2]


    def replicate(self, xreps: int, yreps: int, zreps: int = 1) -> None:
        """
        Replicate upper and lower slabs of the Bicrystal.

        Args:
            xreps (int): Number of repetitions along x-axis.

            yreps (int): Number of repetitions along y-axis.

            zreps (int, optional): Number of repetitions along z-axis.
            Defaults to 1.

        Returns:
            None
        """
        reps = (xreps, yreps, zreps)
        self.lower *= reps
        self.upper *= reps
        self.rxyz = reps


    def shift_upper(self, xshift: float, yshift: float, zshift: float = 0.0) -> None:
        """
        Translate atom positions in the upper slab only.

        Args:
            xshift (float): Translation along x-axis.

            yshift (float): Translation along y-axis.

            zshift (float, optional): Translation along z-axis.
            Defaults to 0.0.

        Returns:
            None
        """
        shifts = [xshift, yshift, zshift]
        self.upper.positions += shifts
        self.dxyz = shifts


    def get_bounds(self, algo: dict) -> None:
        """
        Get the bounds for the GB region for LAMMPS input.

        Args:
            algo (dict): Dictionary of algorithm parameters.

        Returns:
            None
        """
        lowerb = self.lower.cell[2, 2] - algo["gb_thick"]   # distance from the bottom (intuitive)
        # upperb = lowerb + 2 * algo["gb_thick"] + algo["gb_gap"]   # distance from the bottom
        upperb = self.upper.cell[2, 2] - algo["gb_thick"] # distance from the top (account for MD)
        self.bounds = [lowerb, upperb, algo["pad"]]


    def get_gbplane_atoms_u(self) -> int:
        """
        Get the number of GB atoms in one plane of the upper slab.

        Returns:
            npp_u (int): Number of atoms in the lowest plane(s).
        """
        sorted_pos = self.upper.positions[self.upper.positions[:, 2].argsort()]
        min_top = sorted_pos.round(6)[0, 2]
        mask = self.upper.positions[:, 2] < (min_top + Z_THRESH)
        if self.dlat > 0:
            if self.debug: print(f"Calculating Nplane atoms within {self.dlat:.6f} A")
            mask = self.upper.positions[:, 2] < (min_top + self.dlat - Z_THRESH)
        else:
            print(f"dlat is given as {self.dlat} <= 0, taking planar slice.")
        self.npp_u = sum(mask)
        if self.debug: print(f"Atoms per top crystal plane = {self.npp_u}\n")

        self.gbplane_ids_u = np.where(mask)[0]
        self.gbplane_pos_u = self.upper.positions[self.gbplane_ids_u]
        return self.npp_u


    def make_vacancies_u(self, index_list: list) -> None:
        """
        Remove atoms from upper at the given indices.

        Args:
            index_list (list): List of atom indices to remove from the slab.
        """
        index_sorted = sorted(index_list, reverse=True)
        for i,idx in enumerate(index_sorted):
            self.upper.pop(idx)


    def defect_upper(self, algo: dict, rng: np.random.Generator) -> None:
        """
        Create vacancies on the bottom plane of the upper crystal.

        Args:
            algo (dict): Dictionary of algorithm parameters.

            rng (np.random.Generator): Random number generator from NumPy.

        Raises:
            Assertion: Number of atoms per plane must be calculated first.

            Assertion: Fraction of atoms remaining must be consistent.

        Returns:
            None.
        """
        assert self.npp_u is not None, "Must calculate the number of atoms " + \
            "per plane first! Call get_gbplane_atoms_u()"

        n_vac = int(rng.integers(np.floor(self.npp_u * (1 - algo["frac_max"])),
                                 np.ceil(self.npp_u * (1 - algo["frac_min"])),
                                 endpoint=True))

        to_delete = rng.choice(self.gbplane_ids_u, size=n_vac, replace=False)
        self.make_vacancies_u(to_delete)
        n_udef = self.upper.get_global_number_of_atoms()
        if self.debug: print(f"{n_udef} atoms in defective cell due to {n_vac} vacancies created.\n")

        # recompute ngb from the new cell
        self.n = np.mod(n_udef, self.npp_u)
        assert self.n == np.mod(self.npp_u - n_vac, self.npp_u), "\n~ ERROR!!! ~\n" + \
            "Atoms were not deleted properly!\n" + \
            f"{n_udef} atoms in defective cell due to {n_vac} vacancies created.\n" + \
            f"And with {self.npp_u} atoms/plane, a value of ngb={self.n} was calculated, " + \
            f"which doesn't equal {np.mod(self.npp_u - n_vac, self.npp_u)}."

        self.n /= self.npp_u


    def perturb_atoms(self, rng: np.random.Generator) -> None:
        """
        Randomly perturb the atoms near the GB.

        Args:
            rng (np.random.Generator): Random number generator from NumPy.

        Returns:
            None.
        """
        mask_upper = self.upper.positions[:,2] < self.algo["gb_thick"] / 2
        N_u = sum(mask_upper)
        self.upper.positions[mask_upper, :] += self.dupper * rng.random([N_u, 3])

        mask_lower = self.lower.positions[:,2] > self.lower.cell[2,2] - self.algo["gb_thick"] / 2
        N_l = sum(mask_lower)
        self.lower.positions[mask_lower, :] += self.dlower * rng.random([N_l, 3])


    def join_gb(self, algo: dict, gb_normal: int = 2) -> None:
        """
        Join upper and lower slabs to create GB structure.

        Args:
            algo (dict): Dictionary of algorithm parameters.

            gb_normal (int, optional): Index of axis normal to GB in Bicrystal.
            Defaults to 2.

        Returns:
            None.
        """
        offset = self.lower.cell[gb_normal, gb_normal] + algo["gb_gap"]
        upper_copy = self.upper.copy()   # upper parent is unchanged
        upper_copy.positions[:, gb_normal] += offset
        upper_copy.extend(self.lower.copy())   # copy lower parent to be safe?
        upper_copy.cell[gb_normal, gb_normal] += offset + algo["vacuum"]
        self.gb = upper_copy


    def write_gb(self, filename: str) -> None:
        """
        Write the joined GB structure to a LAMMPS data file.

        Args:
            filename (str): Filename of GB structure.

        Returns:
            None, but a lammps-data file is written.
        """
        assert self.gb is not None, "GB hasn't been created yet! Use the " + \
            "join_gb() method before calling write_gb()."

        write_lammps_data(filename, self.gb)


    def get_edge_midpts(self, pts: list, rv: list) -> np.ndarray:
        """
        Compute the midpoints of Voronoi edges.

        Args:
            pts (list): List of Voronoi vertices.

            rv (list): List of Voronoi edges with vertex indices.

        Returns:
            A NumPy array of edge midpoints.
        """
        emps = []
        for edge in rv:
            emps.append((np.array(pts[edge[0]]) + np.array(pts[edge[1]])) / 2)
        return np.array(emps)


    def compute_voronoi(self, struct0: Lattice, bounds: tuple, edge_midpoints: bool,
                        reps: np.ndarray=np.array([3, 3, 1])) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the Voronoi diagram for the 3D structure.

        Args:
            struct0 (Lattice): Input structure with GB.

            bounds (tuple): Upper and lower bounds for the GB region.

            edge_midpoints (bool): Whether to include edge midpoints.

            reps (np.ndarray, optional): Number of replications to include points near surfaces (pbc).
            Default is np.array([3, 3, 1]).

        Returns:
            v (np.ndarray): Voronoi vertices

            pts (np.ndarray): Atomic positions.
        """
        a, b, c = struct0.cell.lengths()
        struct = struct0 * reps
        pts = struct.positions - (reps // 2) * np.array([a, b, c])

        vor = Voronoi(pts)
        if self.debug: print(f"Creating Voronoi diagram with {reps} repetitions.")

        vert = np.round(vor.vertices, 6)
        if self.debug: print(f"Found {len(vert)} total vertices.")

        if edge_midpoints:
            rv_pos = [rv for rv in vor.ridge_vertices if (np.array(rv) >= 0).all()]
            emps = self.get_edge_midpts(vert, rv_pos)
            vert = np.concatenate((vert, emps), axis=0)
            if self.debug: print(f"Adding {len(emps)} edge midpoints.")

        # Keep only the sites within the GB region
        mask = (vert[:, 0] > 0) & (vert[:, 0] < a) & \
               (vert[:, 1] > 0) & (vert[:, 1] < b) & \
               (vert[:, 2] > bounds[0]) & (vert[:, 2] < bounds[1])
        v = vert[mask]
        v = v[v[:, 2].argsort()]   # easier to identify
        if self.debug: print(f"After masking, {len(v)} points were returned.\n")
        return v, pts


    def check_exist(self, exist: list, curr: np.ndarray, n1: int,
                    tol: float=None) -> bool:
        """
        Check if a NumPy array already exists in a list.

        Args:
            exist (list): Existing list of points.

            curr (np.ndarray): Current array to check.

            n1 (int): Number of elements to compare.

            tol (float, optional): Tolerance. Default is None.

        Returns:
            Truth value of existence.
        """
        if not tol: tol = min(curr[:n1]) * 2e-2
        diff = np.array([np.linalg.norm(x[:n1] - curr[:n1]) for x in exist])
        # if self.debug: print("tol, diff:", tol, diff)
        if len(diff) == 0:
            exist.append(curr)
            return False
        elif (diff > tol).all():
            exist.append(curr)
            return False
        return True


    def classify_sites(self, sites: np.ndarray, pos: np.ndarray, top_n: int=10,
                       abs_tol: float=6e-1, rel_tol: float=5e-3,
                       reset_index: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Classifies intersitial sites based on conventional octahedral, tetrahedral, etc.

        Args:
            sites (np.ndarray): STUB

            pos (np.ndarray): STUB

            top_n (int, optional): STUB
            Default is 10.

            abs_tol (float, optional): STUB
            Default is 6e-1.

            rel_tol (float, optional): STUB
            Default is 5e-3.

            reset_index (bool, optional): Reset DataFrame indices.
            Default is True.

        Returns:
            df (pd.DataFrame): DataFrame of all interstial sites.

            unique (pd.DataFrame): DataFrame of all unique interstitial sites.
        """
        labels = []
        nn_list = []
        nnd_list = []
        other = []
        tc, oc, tstrainc, ostrainc, trc, trstrainc, otherc = [0] * 7
        for i,v in enumerate(sites):
            l = np.sort(np.linalg.norm(pos - v, axis=1))[:top_n]   # top_n shortest atom distances to site
            vdist = np.abs(l - l[0])   # compare each of top_n atoms dist w/ the shortest dist

            nn1 = sum(vdist < abs_tol)          # num neighbors < abs_tol from closest neighbor
            nn2 = sum(vdist < rel_tol * l[0])   # num neighbors < rel_tol * shortest dist

            nn1_dist = sum(vdist[:nn1])  # add up distances to the nn1 neighbors
            same_dist = nn1_dist < rel_tol * l[0]  # all nn1 are ~same distance away

            nn_list.append(nn2)  # track the number of true nearest neighbors
            nnd_list.append(l[:nn1].round(6))  # track the positions of nn1

            # These are for labeling purposes ONLY
            exist = self.check_exist(other, l, nn1, tol=None)   # which tol is best?
            if nn2 == 3:
                if same_dist:
                    if not exist: trc += 1
                    labels.append(f'triangular{trc}')
                else:
                    if not exist: tstrainc += 1
                    labels.append(f'tri_strain{trstrainc}')
            elif nn2 == 4:
                if same_dist:
                    if not exist: tc += 1
                    labels.append(f'tetrahedral{tc}')
                else:
                    if not exist: tstrainc += 1
                    labels.append(f'tetra_strain{tstrainc}')
            elif nn2 == 6:
                if same_dist:
                    if not exist: oc += 1
                    labels.append(f'octahedral{oc}')
                else:
                    if not exist: ostrainc += 1
                    labels.append(f'octa_strain{ostrainc}')
            else:
                if not exist: otherc += 1
                labels.append(f'other{otherc}')

            if self.debug and i == 0:
                print(f"For vertex {i}")
                print('l:', l)
                print('vdist:', vdist)
                print('nn1:', nn1)
                print('nn2:', nn2)
                print('nn1_dist:', nn1_dist)
                print()

        df = pd.DataFrame({'x':sites[:, 0], 'y':sites[:, 1], 'z':sites[:, 2],
                           'label':labels, 'nn': nn_list, 'nnd':nnd_list})
        df.sort_values(by=['z', 'y', 'x'], ignore_index=True, inplace=True)

        unique = df.drop_duplicates(subset=['label', 'nn'], keep='first')
        if reset_index: unique.reset_index(drop=True, inplace=True)

        return df, unique


    def find_interstitials(self, zbounds: list=None, edges: bool=False,
                           unique_sites: bool=False) -> list:
        """
        Finds interstitial positions in the GB region of a Bicrystal.

        Args:
            zbounds (list, optional): Top and lower bounds of the GB region.
            Defaults to None.

            edges (bool, optional): Whether to include midpoints of Voronoi edges.
            Defaults to False.

            unique_sites(bool, optional): Filter out duplicate sites.
            Defaults to False.

        Returns:
            A list of Interstitial objects.
        """
        assert self.gb is not None, "GB hasn't been created yet! Use the " + \
            "join_gb() method before calling find_interstitials()."

        if not zbounds: zbounds = (self.bounds[0], self.z - self.bounds[1])
        if self.debug: print(f"Searching for interstitials between {zbounds}.")
        v, pts = self.compute_voronoi(self.gb, zbounds, edges)

        df, unique = self.classify_sites(v, pts)

        if unique_sites:
            self.interstitials = Interstitial.from_df(unique)
        else:
            self.interstitials = Interstitial.from_df(df)

        return self.interstitials


    def swap_gb_interstitials(self, zbounds: list) -> int:
        """
        Swap atoms in GB region with intersitial positions.

        Args:
            zbounds (list, optional): Top and lower bounds of the region
            for swapping GB atoms.

        Returns:
            swapped_n (int): The number of GB atoms that were swapped.
        """
        # Get GB atoms
        gb_mask = (self.gb.positions[:, 2] >= zbounds[0]) & \
                  (self.gb.positions[:, 2] <= zbounds[1])
        gb_ind = np.where(gb_mask)[0]
        np.random.shuffle(gb_ind)

        if len(gb_ind) < self.algo["inter_n"]:
            if len(gb_ind) <= len(self.interstitials):
                print(f"Warning: Only {len(gb_ind)} GB atoms to swap instead of {self.algo['inter_n']}.")
            else:
                print(f"Warning: Only {len(self.interstitials)} interstitial sites to swap instead of {self.algo['inter_n']}.")

        candidates = [self.algo['inter_n'], len(self.interstitials), len(gb_ind)]
        if self.debug: print(candidates)
        swapped_n = min(candidates)
        for i in range(swapped_n):
            self.gb[gb_ind[i]].position = self.interstitials[i].position()

        return swapped_n


    def find_and_swap_inters(self, rng) -> int:
        """
        Find interstitial sites near GB and swap with GB atoms.

        Args:
            rng (np.random.Generator): Random number generator from NumPy.

        Returns:
            swapped_n (int): The number of GB atoms that were swapped.
        """
        if self.algo["inter_n"] > 0 and rng.random() < self.algo["inter_p"]:
            zmid = self.lower0.cell[2, 2] + self.algo["gb_gap"] / 2
            zbounds = [zmid - self.algo["inter_t"], zmid + self.algo["inter_t"]]
            inters = self.find_interstitials(zbounds=zbounds, unique_sites=self.algo["inter_u"])
            if self.algo['inter_r']:
                random.shuffle(inters)
            if self.debug: print(f"Found {len(inters)} interstitial sites.")
            if self.debug: print(f"The first of which is: {inters[0]}\n")

            zbounds2 = [zmid - 2 * self.algo["inter_t"], zmid + 2 * self.algo["inter_t"]]
            swapped_n = self.swap_gb_interstitials(zbounds=zbounds2)
            return swapped_n
        else:
            return 0


