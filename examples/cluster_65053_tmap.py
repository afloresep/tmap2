"""TMAP of cluster_65053 with all chemistry utilities."""

import csv

from tmap import TMAP
from tmap.utils.chemistry import (
    AVAILABLE_PROPERTIES,
    fingerprints_from_smiles,
    molecular_properties,
    murcko_scaffolds,
)


def main() -> None:
    smiles = []
    with open("examples/cluster_65053.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles.append(row["smiles"])

    print(f"Loaded {len(smiles):,} SMILES")

    fps, valid_idx = fingerprints_from_smiles(smiles, fp_type="morgan", n_bits=2048)
    valid_smiles = [smiles[i] for i in valid_idx]

    model = TMAP(n_neighbors=20).fit(fps)

    # --- Compute ALL molecular properties ---
    props = molecular_properties(valid_smiles, properties=list(AVAILABLE_PROPERTIES))

    # --- Compute Murcko scaffolds ---
    scaffolds = murcko_scaffolds(valid_smiles)

    # --- Build visualization ---
    viz = model.to_tmapviz()
    viz.title = "Cluster 65053 — Chemistry Utils Demo"

    # Add SMILES for structure rendering in tooltip
    viz.add_smiles(valid_smiles)

    # Add every numeric property as a color layout
    for name, values in props.items():
        viz.add_color_layout(name, values)

    # Add scaffolds as a tooltip label (too many unique values for coloring)
    viz.add_label("murcko_scaffold", scaffolds.tolist())

    # --- Save HTML + serve ---
    out = viz.write_html("examples/cluster_65053_tmap.html")
    print(f"Saved to {out}")

    viz.serve(port=8050)


if __name__ == "__main__":
    main()
