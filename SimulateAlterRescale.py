import argparse
import os
import random
from dendropy.simulate import treesim
from ete3 import Tree
import numpy.random as npr
from utils import read_list
from numpy.random import normal
from multiprocessing import Pool
import multiprocessing
from functools import partial
from contextlib import contextmanager

ROOT = os.path.dirname(__file__)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def scaleBranch(list_of_rates, list_of_times):
    new_length = 0.0
    for i in range(1, len(list_of_times)):
        new_length += list_of_rates[i - 1] * (list_of_times[i] - list_of_times[i - 1])
    return new_length


def rescale_tree(t, scale):
    diam = tree_diam(t)
    for node in t.traverse("postorder"):
        if node.is_root():
            pass
        else:
            node.dist = node.dist * scale / diam
    return t


def tree_diam(t):
    distances = {}
    for i, leaf1 in enumerate(t):
        if i == 0:
            A = leaf1
        distances[(A, leaf1)] = A.get_distance(leaf1)
    B = max(distances, key=distances.get)[1]
    distances = {}
    for i, leaf1 in enumerate(t):
        distances[(B, leaf1)] = B.get_distance(leaf1)
    diam = max(distances.values())
    return diam


def sample_scale(dms):
    mean = random.sample(dms, 1)[0]
    scale = normal(loc=mean, scale=mean / 10)
    if scale > 0.02:
        return scale
    else:
        return 0.02


def simulateTree(
    i,
    numtips,
    treeType,
    outdir,
    verbose,
    diams,
    use_bl,
    rate_small,
    multiplier_small,
    rate_big,
    multiplier_big,
    minimum_value,
):
    scale = sample_scale(diams)
    outname = os.path.join(outdir, str(i) + "_" + str(numtips) + "_tips.nwk")
    if treeType == "birth-death":  # using dendropy
        t = treesim.birth_death_tree(
            birth_rate=1.0, death_rate=0.5, num_extant_tips=numtips
        )
        t.write(path=outname, schema="newick", suppress_rooting=True)
        t = Tree(outname)
        rates = dict()
        rates[t.get_tree_root()] = 1.0
        number_of_small_changes_per_branch = dict()
        number_of_big_changes_per_branch = dict()

        average_dist = 0.0
        n_branches = 0

        for n in t.traverse(strategy="preorder"):
            if n != t.get_tree_root():
                d = n.dist
                average_dist += d
                n_branches += 1

        average_dist = average_dist / n_branches

        for n in t.traverse(strategy="preorder"):
            if n != t.get_tree_root():
                d = 1.0
                normalizing_factor = 1.0
                if use_bl:
                    d = n.dist
                    normalizing_factor = 1.0
                else:
                    d = average_dist
                    normalizing_factor = n.dist / average_dist
                event_time = 0.0
                list_of_rates = list()
                list_of_times = list()
                list_of_rates.append(rates[n.up])
                list_of_times.append(0.0)
                latest = 0.0
                rate_multiplier = 1.0
                number_of_small_changes_per_branch[n] = 0
                number_of_big_changes_per_branch[n] = 0
                while event_time < d:
                    t_small = npr.exponential(scale=rate_small)
                    t_big = npr.exponential(scale=rate_big)
                    if t_small < t_big:
                        event_time = latest + t_small
                        rate_multiplier = npr.lognormal(mean=0, sigma=multiplier_small)
                        number_of_small_changes_per_branch[n] += 1
                    else:
                        event_time = latest + t_big
                        rate_multiplier = npr.lognormal(mean=0, sigma=multiplier_big)
                        number_of_big_changes_per_branch[n] += 1
                    if event_time < d:
                        list_of_times.append(event_time * normalizing_factor)
                        latest_rate = list_of_rates[-1]
                        list_of_rates.append(latest_rate * rate_multiplier)
                    latest = event_time
                list_of_times.append(d * normalizing_factor)
                # We've generated all the change points along the branch, let's scale the branch accordingly.
                new_branch_length = scaleBranch(list_of_rates, list_of_times)
                if verbose:
                    print(
                        "Number of small changes on branch of length "
                        + str(n.dist)
                        + " : "
                        + str(number_of_small_changes_per_branch[n])
                    )
                    print(
                        "Number of big changes on branch of length "
                        + str(n.dist)
                        + " : "
                        + str(number_of_big_changes_per_branch[n])
                    )
                    print(
                        "\t\tOld vs new distance: "
                        + str(n.dist)
                        + " <-> "
                        + str(new_branch_length)
                        + "\n"
                    )
                n.dist = new_branch_length
                # Let's set the rate for the current node
                rates[n] = list_of_rates[-1]

        # Additional tree traversal: we do not want branch lengths under some value.
        # blAfter=list()
        t = rescale_tree(t, scale=scale)
        for n in t.traverse(strategy="preorder"):
            if n.is_root():
                pass
            else:
                if n.dist < minimum_value and n.is_leaf():
                    while n.dist < minimum_value:
                        n.dist = normal(loc=minimum_value, scale=0.005)
                        # n.dist=npr.exponential(scale=minimum_value)
        #        blAfter.append(n.dist)
        # t=rescale_tree(t,scale=scale)
        t.write(format=1, outfile=outname)

    elif treeType == "uniform":  # using ete3
        t = Tree()
        t.populate(numtips)
        t = rescale_tree(t, scale=scale)
        t.write(format=1, outfile=outname)
    else:
        exit("Error, treetype should be birth-death or uniform")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntips", type=int, default=20, help="how many tips")
    parser.add_argument("--ntrees", type=int, help="how many trees")
    parser.add_argument("--type", type=str, help="birth-death or uniform")
    parser.add_argument(
        "--o",
        type=str,
        default="",
        help="path to the output directory were the\
    .nwk tree files will be saved",
    )
    parser.add_argument("--verbose", type=str, default="false")

    args = parser.parse_args()
    numtips = args.ntips
    numtrees = args.ntrees
    treeType = args.type
    outdir = args.o
    verbose = args.verbose == "true"

    # 0.03 0.1 1.0 0.2 0.01
    use_bl = False
    rate_small = 0.03  # scale parameter of small changes
    multiplier_small = 0.1  # variance of lognormal distribution from which are drawn small rate multipliers
    rate_big = 1.0  # scale parameter of large changes
    multiplier_big = 0.2  # variance of lognormal distribution from which are drawn large rate multipliers
    minimum_value = 0.001  # minimum allowed branch length

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    hodiams = read_list(os.path.join(ROOT, "hogenom_diams.txt"))
    raxdiams = read_list(os.path.join(ROOT, "raxml_diams.txt")) * 10
    diams = hodiams + raxdiams
    diams = [item for item in diams if item > 0.02 and item < 15]

    for i in range(numtrees):
        simulateTree(
            i,
            numtips=numtips,
            treeType=treeType,
            outdir=outdir,
            verbose=verbose,
            diams=diams,
            use_bl=use_bl,
            rate_small=rate_small,
            multiplier_small=multiplier_small,
            rate_big=rate_big,
            multiplier_big=multiplier_big,
            minimum_value=minimum_value,
        )

    # with poolcontext() as pool:
    #    results = pool.map(partial(simulateTree, numtips=numtips,treeType=treeType,outdir=outdir,verbose=verbose,diams=diams, use_bl = use_bl,
    # rate_small = rate_small, multiplier_small = multiplier_small,rate_big = rate_big, multiplier_big = multiplier_big, minimum_value = minimum_value), range(numtrees))
