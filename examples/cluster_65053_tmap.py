"""Build a molecule TMAP from examples/cluster_65053.csv.

This is the clearest chemistry example in the repo:

1. Load SMILES from a CSV
2. Compute Morgan fingerprints
3. Compute a few molecular properties and Murcko scaffolds
4. Fit `TMAP(metric="jaccard")`
5. Explore the result in HTML or with `serve()`

Usage:
    python examples/cluster_65053_tmap.py
    python examples/cluster_65053_tmap.py --nrows 3000
    python examples/cluster_65053_tmap.py --serve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tmap import TMAP
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

DATA_PATH = Path(__file__).with_name("cluster_65053.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("cluster_65053.html")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a TMAP for cluster_65053 molecules.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=0,
        help="Number of rows to load from the CSV. Use 0 for all rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML output path.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server after building the visualization.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for --serve.",
    )
    return parser


def load_smiles(nrows: int) -> list[str]:
    read_rows = None if nrows == 0 else nrows
    df = pd.read_csv(DATA_PATH, nrows=read_rows)
    return df["smiles"].tolist()


def main() -> None:
    args = build_parser().parse_args()

    smiles = load_smiles(args.nrows)
    print(f"Loaded {len(smiles):,} molecules from {DATA_PATH.name}")

    print("Computing Morgan fingerprints...")
    fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)

    print("Computing molecular properties...")
    props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed"])

    print("Computing Murcko scaffolds...")
    scaffolds = murcko_scaffolds(smiles)

    print("Fitting TMAP...")
    model = TMAP(
        metric="jaccard",
        n_neighbors=20,
        n_permutations=512,
        kc=50,
        seed=42,
    ).fit(fps)

    viz = model.to_tmapviz()
    viz.title = "Cluster 65053"
    viz.add_smiles(smiles)
    viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
    viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
    viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
    viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
    viz.add_label("Murcko Scaffold", scaffolds.tolist())

    output_path = viz.write_html(args.output)
    print(f"Saved HTML to {output_path}")

    if args.serve:
        print(f"Serving on http://127.0.0.1:{args.port}")
        viz.serve(port=args.port)


if __name__ == "__main__":
    main()
