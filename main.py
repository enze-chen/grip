#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np

from gbgrip import run

##############################################################################

if __name__ == "__main__":
    parser = ArgumentParser(description="Perform grand canonical optimization of GBs.")
    parser.add_argument("-i", "--input", type=str, default="params.yaml",
                        help="File containing structure & algorithm parameters.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run in DEBUG mode, which prints variables and terminates early.")
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Random seed for debugging.")
    args = parser.parse_args()
    infile = args.input
    debug = args.debug
    seed = args.seed

    run(infile, debug, seed)
