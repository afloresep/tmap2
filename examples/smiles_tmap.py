"""
End-to-end TMAP visualization from SMILES strings.

This script demonstrates the full pipeline:
    SMILES -> Morgan Fingerprints -> MinHash -> LSHForest -> Layout -> Visualization

Requirements:
    pip install rdkit tqdm

Usage:
    python examples/smiles_tmap.py
    # Or with custom SMILES list:
    from examples.smiles_tmap import create_tmap_from_smiles
    coords = create_tmap_from_smiles(my_smiles_list, "output.html")
"""

import numpy as np
from tqdm import tqdm

# Check RDKit availability
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig, ScalingType
from tmap.visualization import TmapViz


def smiles_to_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[np.ndarray, list[int], list]:
    """
    Convert SMILES strings to Morgan fingerprints.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius (default 2 = ECFP4)
        n_bits: Number of bits in fingerprint

    Returns:
        fingerprints: numpy array of shape (n_valid, n_bits)
        valid_indices: indices of successfully parsed SMILES
        mols: list of RDKit molecule objects
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required: pip install rdkit")

    # Use the newer MorganGenerator API
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    fingerprints = []
    valid_indices = []
    mols = []

    for i, smi in tqdm(enumerate(smiles_list), desc='Generating Fingerprints', total=len(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = morgan_gen.GetFingerprintAsNumPy(mol)
            fingerprints.append(fp.astype(np.uint8))
            valid_indices.append(i)
            mols.append(mol)

    return np.array(fingerprints), valid_indices, mols


def compute_molecular_properties(mols: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute molecular properties for coloring."""
    mw = [Descriptors.MolWt(mol) for mol in mols] # type: ignore
    logp = [Descriptors.MolLogP(mol) for mol in mols] # type: ignore
    n_rings = [rdMolDescriptors.CalcNumRings(mol) for mol in mols]
    return np.array(mw), np.array(logp), np.array(n_rings)


def create_tmap_from_smiles(
    smiles: list[str],
    output_path: str = "tmap_molecules.html",
    title: str = "TMAP Visualization",
    k: int = 20,
    kc: int = 100,
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
        kc: Search quality multiplier
        num_perm: Number of MinHash permutations
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x, y) coordinates
    """
    print(f"Processing {len(smiles)} SMILES...")

    # Step 1: Convert SMILES to fingerprints
    print("Step 1: Computing Morgan fingerprints...")
    fingerprints, valid_indices, mols = smiles_to_fingerprints(smiles)
    n = len(valid_indices)
    print(f"  Valid molecules: {n}/{len(smiles)}")

    if n < 2:
        raise ValueError("Need at least 2 valid molecules")

    # Get valid SMILES
    valid_smiles = [smiles[i] for i in valid_indices]

    # Step 2: Compute molecular properties
    print("Step 2: Computing molecular properties...")
    mw, logp, n_rings = compute_molecular_properties(mols)

    # Step 3: MinHash encoding
    print("Step 3: Encoding with MinHash...")
    mh = MinHash(num_perm=num_perm, seed=seed)
    signatures = mh.batch_from_binary_array(fingerprints)
    print(f"  Signature shape: {signatures.shape}")

    # Step 4: Build LSH Forest
    print("Step 4: Building LSH Forest...")
    lsh = LSHForest(d=num_perm, l=64)
    lsh.batch_add(signatures)
    lsh.index()

    # Step 5: Compute layout
    print("Step 5: Computing layout...")

    cfg = LayoutConfig()
    cfg.k = k
    cfg.kc = kc
    cfg.node_size = 1/30
    cfg.mmm_repeats = 2
    cfg.sl_extra_scaling_steps = 10
    cfg.sl_scaling_type = ScalingType.RelativeToDrawing
    cfg.fme_iterations = 1000
    cfg.deterministic = True
    cfg.seed = seed

    x, y, s, t = layout_from_lsh_forest(lsh, cfg)

    print(f"  Nodes: {len(x)}")
    print(f"  Edges: {len(s)} (fully connected tree has {n-1} edges)")

    # Step 6: Create visualization
    print("Step 6: Creating visualization...")
    viz = TmapViz()
    viz.title = title
    viz.background_color = "#FFFFFF"
    viz.point_color = "#4a9eff"
    viz.point_size = 4.0
    viz.opacity = 0.9

    # Set coordinates
    viz.set_points(x, y)

    # Add SMILES for structure rendering
    viz.add_smiles("SMILES", valid_smiles)

    # Add molecular properties as color layouts
    viz.add_color_layout("Molecular Weight", mw.tolist(), categorical=False, color="viridis")
    viz.add_color_layout("LogP", logp.tolist(), categorical=False, color="plasma")
    viz.add_color_layout("Number of Rings", n_rings.tolist(), categorical=False, color="coolwarm")

    # Add index labels
    viz.add_label("Index", [str(i) for i in range(n)])

    # Save visualization
    if not output_path.endswith('.html'):
        output_path = output_path + '.html'

    html = viz.render()
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nSaved visualization to: {output_path}")
    return x, y


# Example SMILES - common drug-like molecules
import pandas as pd
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent
csv_path = script_dir / 'cluster_65053.csv'
df = pd.read_csv(csv_path)
EXAMPLE_SMILES = df.smiles.to_list()

if __name__ == "__main__":
    if not RDKIT_AVAILABLE:
        print("RDKit is required for this example.")
        print("Install with: pip install rdkit")
        exit(1)

    # Create more data by adding variations
    smiles_list = EXAMPLE_SMILES.copy()

    # Add some simple modifications to increase dataset size
    for smi in EXAMPLE_SMILES[:15]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # Add methylated versions
            smiles_list.append(smi.replace("c1", "Cc1", 1))
            # Add hydroxylated versions
            smiles_list.append(smi.replace("c1", "Oc1", 1))

    # Filter valid SMILES
    valid_smiles = [s for s in smiles_list if Chem.MolFromSmiles(s) is not None]
    print(f"Working with {len(valid_smiles)} molecules")

    # Create visualization
    coords = create_tmap_from_smiles(
        smiles=valid_smiles,
        output_path="tmap_molecules.html",
        title="Molecular TMAP Demo",
        k=20,
        num_perm=128,
        seed=42,
    )

    print(f"\nDone! Open the HTML file in a browser to view the interactive visualization.")
    print("  - Pan: Click and drag")
    print("  - Zoom: Scroll wheel")
    print("  - Hover: See molecule structure and properties")
    print("  - Use dropdown to change color scheme")
