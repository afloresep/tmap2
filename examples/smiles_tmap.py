"""
End-to-end TMAP visualization from SMILES strings.

This script demonstrates the full pipeline:
    SMILES -> Morgan Fingerprints -> MinHash -> LSHForest -> Layout -> Visualization

Two layout approaches are available:
1. layout_from_lsh_forest (RECOMMENDED) - passes full kNN graph to OGDF for MST
2. MSTBuilder + ForceDirectedLayout - computes MST in Python first

The first approach typically produces better connected trees because OGDF
computes the MST on the full kNN graph rather than a pre-filtered version.

Requirements:
    pip install rdkit
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from tqdm import tqdm
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig, ScalingType
from tmap.visualization.tmapviz import TmapViz


def smiles_to_fingerprints(smiles_list: list[str], radius: int = 2, n_bits: int = 2048):
    """
    Convert SMILES strings to Morgan fingerprints.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default 2 = ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        fingerprints: numpy array of shape (n_samples, n_bits)
        valid_indices: indices of successfully parsed SMILES
        mols: list of RDKit molecule objects
    """
    from rdkit.Chem import rdFingerprintGenerator

    # Use the newer MorganGenerator API (avoids deprecation warnings)
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = []
    valid_indices = []
    mols = []

    for i, smi in tqdm(enumerate(smiles_list), desc='Generating Fingerprints...', total=len(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp.astype(np.uint8))
            valid_indices.append(i)
            mols.append(mol)

    return np.array(fingerprints), valid_indices, mols


def compute_molecular_properties(mols: list):
    """Compute molecular properties for coloring."""
    mw = [Descriptors.MolWt(mol) for mol in mols]
    logp = [Descriptors.MolLogP(mol) for mol in mols]
    number_of_rings = [rdMolDescriptors.CalcNumRings(mol) for mol in mols]
    return np.array(mw), np.array(logp), np.array(number_of_rings)


def create_tmap_from_smiles(
    smiles: list[str],
    output_path: str = "tmap_output.html",
    title: str = "TMAP Visualization",
    k: int = 10,
    num_perm: int = 128,
    seed: int = 42,
):
    """
    Create a TMAP visualization from SMILES strings.

    Args:
        smiles: List of SMILES strings
        output_path: Path to save HTML file
        title: Title for the visualization
        k: Number of nearest neighbors for k-NN graph
        num_perm: Number of MinHash permutations
        seed: Random seed for reproducibility
    """
    print(f"Processing {len(smiles)} SMILES...")

    # Step 1: Convert SMILES to fingerprints
    print("Computing Morgan fingerprints...")
    fingerprints, valid_indices, mols = smiles_to_fingerprints(smiles)
    n = len(valid_indices)
    print(f"  Valid molecules: {n}/{len(smiles)}")

    if n < 2:
        raise ValueError("Need at least 2 valid molecules")

    # Get valid SMILES
    valid_smiles = [smiles[i] for i in valid_indices]

    # Step 2: Compute molecular properties
    print("Computing molecular properties...")
    mw, logp, n_rings = compute_molecular_properties(mols)

    # Step 3: MinHash encoding
    print("Encoding with MinHash...")
    mh = MinHash(num_perm=num_perm, seed=seed)
    signatures = mh.batch_from_binary_array(fingerprints)
    print(f"  Signature shape: {signatures.shape}")

    # Step 4: Build LSH Forest
    print("Building LSH Forest...")
    lsh = LSHForest(d=num_perm, l=64)  # More trees for better recall
    lsh.batch_add(signatures)
    lsh.index()

    # Step 5: Compute layout directly from LSHForest (RECOMMENDED)
    # This passes the full kNN graph to OGDF, which computes MST internally.
    # This approach typically produces better connected trees.
    print("Computing layout (passes kNN graph to OGDF for MST)...")

    k_actual = min(k, n - 1)

    # Configure layout - matches old TMAP parameters
    cfg = LayoutConfig()
    cfg.k = k_actual
    cfg.kc = 50  # Query multiplier for LSH
    cfg.node_size = 1/30  # Affects repulsion
    cfg.mmm_repeats = 2  # More layout iterations
    cfg.sl_extra_scaling_steps = 10
    cfg.sl_scaling_type = ScalingType.RelativeToAvgLength
    cfg.fme_iterations = 500
    cfg.deterministic = True
    cfg.seed = seed

    x, y, s, t = layout_from_lsh_forest(lsh, cfg)

    print(f"  Nodes: {len(x)}")
    print(f"  Edges: {len(s)} (MST has n-1={n-1} for connected, fewer if disconnected)")

    # Create a simple coords object for visualization
    class Coords:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    coords = Coords(x, y)

    # Step 7: Create visualization
    print("Creating visualization...")
    viz = TmapViz()
    viz.title = title
    viz.background_color = "#FFFFFF"
    viz.point_color = "#4a9eff"
    viz.point_size = 4.0
    viz.opacity = 0.9

    # Set coordinates
    viz.set_points(coords.x, coords.y)

    # Add SMILES for structure rendering
    viz.add_smiles("SMILES", valid_smiles)

    # Add molecular properties as color layouts
    viz.add_color_layout("MW", mw.tolist(), categorical=False, color="rainbow")
    viz.add_color_layout("LogP", logp.tolist(), categorical=False, color="rainbow")
    viz.add_color_layout("Number of Rings", n_rings.tolist(), categorical=False, color="rainbow")

    # Add random cluster labels as example categorical data
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, min(10, n // 5 + 1), size=n)
    viz.add_color_layout("Cluster", clusters.tolist(), categorical=True, color="tab10")

    # Add labels
    viz.add_label("Index", [str(i) for i in range(n)])

    # Render and save
    html = viz.render()

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Saved visualization to: {output_path}")
    return coords


if __name__ == "__main__":
    # Example SMILES - common drug molecules
    # Create the visualization
    import pandas as pd
    df = pd.read_csv('cluster_65053.csv')
    coords = create_tmap_from_smiles(
        smiles=df.smiles.to_list(),
        output_path="/Users/afloresep/Downloads/tmap_molecules.html",
        title="Molecular TMAP - Drug-like Compounds",
        k=50,
        num_perm=128,
        seed=42,
    )

    print(f"\nDone! Open the HTML file in a browser to view the interactive visualization.")
    print(f"  - Pan: Click and drag")
    print(f"  - Zoom: Scroll wheel")
    print(f"  - Hover: See molecule structure and properties")
    print(f"  - Use dropdown to change color scheme")
