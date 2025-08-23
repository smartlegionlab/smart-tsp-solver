# --------------------------------------------------------
# Copyright © 2025, A.A. Suvorov
# All rights reserved.
# --------------------------------------------------------
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Comparison of algorithms TSP')
    parser.add_argument('-n', '--n_cities', type=int, default=1000, help='Number of cities')
    parser.add_argument('-s', '--seed', type=int, default=50, help='Seed for generation')
    parser.add_argument('-g', '--generation', type=str, default='cluster',
                        choices=['random', 'cluster', 'circle'], help='City generation method')
    parser.add_argument('-p', '--post_opt', action='store_true', help='Use post-optimization')
    parser.add_argument('-f', '--fast', action='store_true',
                        help='Fast mode (no optimization for angular method))')
    return parser.parse_args()
