from timeit import default_timer as timer
import numpy as np
import tmap as tm
import argparse

def generate_data_new_tmap(n:int):
    np.random.seed
    g_data =[]
    # Generating some random data
    for _ in range(n):
        g_data.append(np.random.randint(low=0, high=2, size=1000))

    return g_data

def generate_data_old_tmap(n:int):
    np.random.seed
    g_data =[]
    # Generating some random data
    for _ in range(n):
        g_data.append(tm.VectorUchar(np.random.randint(low=0, high=2, size=1000)))
    return g_data


def benchmark_new_tmap():
    """ Main function """

    # Use 128 permutations to create the MinHash
    enc = tm.MinHash(128)
    lf = tm.LSHForest(128)

    ns = [10_000, 100_000, 1_000_000]

    for n in ns:
        print(f"\n{'#' * 82}")
        print(f"\t\tResults for {n:,}")
        print(f"{'#' * 82}")
        data = []
        g_data = []
        np.random.seed = 42
        # Generating some random data
        start = timer()
        g_data = generate_data_new_tmap(n)
        print(f"Generating the data took {(timer() - start)}s.")

        # Use batch_from_binary_array to encode the data
        start = timer()
        data = enc.batch_from_binary_array(g_data)
        print(f"Encoding the data took {(timer() - start) }s.")

        # Use batch_add to parallelize the insertion of the arrays
        start = timer()
        lf.batch_add(data)
        print(f"Adding the data took {(timer() - start) }s.")

        # Index the added data
        start = timer()
        lf.index()
        print(f"Indexing took {(timer() - start) }s.")

        # Find the 10 nearest neighbors of the first entry
        start = timer()
        _ = lf.query_linear_scan_by_id(0, 10)
        print(f"The kNN search took {(timer() - start) }s.")


def benchmark_old_tmap():
    """ Main function """

    # Use 128 permutations to create the MinHash
    enc = tm.Minhash(128)
    lf = tm.LSHForest(128)

    ns = [10_000, 100_000, 1_000_000]

    for n in ns:
        print(f"\n{'#' * 82}")
        print(f"\t\tResults for {n:,}")
        print(f"{'#' * 82}")
        data = []
        g_data = []
        np.random.seed = 42
        # Generating some random data
        start = timer()
        g_data = generate_data_old_tmap(n)
        print(f"Generating the data took {(timer() - start)}s.")

        # Use batch_from_binary_array to encode the data
        start = timer()
        data = enc.batch_from_binary_array(g_data)
        print(f"Encoding the data took {(timer() - start) }s.")

        # Use batch_add to parallelize the insertion of the arrays
        start = timer()
        lf.batch_add(data)
        print(f"Adding the data took {(timer() - start) }s.")

        # Index the added data
        start = timer()
        lf.index()
        print(f"Indexing took {(timer() - start) }s.")

        # Find the 10 nearest neighbors of the first entry
        start = timer()
        _ = lf.query_linear_scan_by_id(0, 10)
        print(f"The kNN search took {(timer() - start) }s.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmap', type=str)
    args =parser.parse_args()

    # If old tmap 
    if args.tmap=='old':
        benchmark_old_tmap()

    elif args.tmap =='new':
        benchmark_new_tmap()

    else:
        raise ValueError("Only 'old' or 'new' options are available")
