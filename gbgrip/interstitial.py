from typing import Sequence

import numpy as np

class Interstitial():
    """
    A class for constructing Interstitial site objects.

    Attributes:
        p (np.ndarray): xyz positions of the site.

        symbol (str): Atomic species in the interstitial.

        nn (n): Number of nearest neighbors to the interstitial.

        nnd (list): Distances to neighbors.

        label (str): Name of interstitial (e.g., tetrahedral)
    """

    def __init__(self,
                 p: Sequence[float],
                 symbol: str=None,
                 nn: int=None,
                 nnd: Sequence[float]=None,
                 label: str=None):
        self.p = np.asarray(p)
        self.symbol = symbol
        self.nn = nn
        self.nnd = np.asarray(nnd)
        self.label = label


    @classmethod
    def from_df(cls, df):
        sites = []
        for a in df.itertuples():
            site = cls(p=[a.x, a.y, a.z], nn=a.nn, nnd=a.nnd, label=a.label)
            sites.append(site)
        return sites

    def __repr__(self):
        return f"{self.__class__.__name__}(symbol={self.symbol}, p={self.p})"
    
    def position(self):
        return self.p
