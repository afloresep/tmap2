# Molecule Tutorial

This tutorial walks through a full chemistry workflow:

1. Load SMILES from `examples/cluster_65053.csv`
2. Compute Morgan fingerprints
3. Compute molecular properties and scaffolds
4. Fit a Jaccard TMAP
5. Explore it in Jupyter, save HTML, or serve it locally

## Requirements

```bash
pip install "tmap2[chemistry,notebook]"
```

The `chemistry` extra installs RDKit, which the fingerprint and property helpers need. The `notebook` extra installs jupyter-scatter, which is only needed for notebook widgets.

Run the code from the repository root. The helpers in step 2 start worker processes, so if you paste the code into a script, put it under `if __name__ == "__main__":`. Notebooks don't need that.

## 1. Load A Molecule Table

```python
import pandas as pd

df = pd.read_csv("examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

print(f"{len(smiles):,} molecules")
```

Use a small subset first. Once the workflow is working, remove `nrows=3000` (the full file has about 64,000 molecules).

## 2. Compute Fingerprints, Properties, And Scaffolds

```python
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

fps, valid = fingerprints_from_smiles(
    smiles, fp_type="morgan", radius=2, n_bits=2048, return_valid=True
)

# Only readable SMILES get a fingerprint. Drop the rest, so the properties,
# scaffolds and tooltips still belong to the right molecules.
smiles = [smi for smi, ok in zip(smiles, valid) if ok]

props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed"])
scaffolds = murcko_scaffolds(smiles)
```

`fingerprints_from_smiles` returns a binary matrix for `metric="jaccard"`.
Without `return_valid=True` you get just the matrix, and a single bad SMILES
shifts every row after it in `props` and `scaffolds`.

`molecular_properties` can compute any of the names in `tmap.utils.AVAILABLE_PROPERTIES`, such as `"tpsa"`, `"hbd"`, `"hba"` or `"fraction_csp3"`.

## 3. Fit TMAP

```python
from tmap import TMAP

model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(fps)

print(model.embedding_.shape)
print(model.tree_.edges.shape)
```

For a binary matrix like this one, TMAP finds neighbors with a USearch HNSW index on the raw bits. The index is built on several threads, so two runs can give slightly different maps. Pass `reproducible=True` if you need identical coordinates every time; the build gets slower.

## 4. Build A Visualization

```python
viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

This gives you:

- molecule structures in tooltips
- multiple property views
- scaffold labels in the tooltip

The [visualization guide](visualization_guide.md) covers filters, search, pinned cards and links.

## 5. Explore In A Notebook

```python
viz.show(width=1000, height=650, controls=True)
```

Use this when you want to switch color layers, filter categories, and inspect points interactively in Jupyter.

Note: notebook widgets do not draw tree edges yet, and they warn about it. HTML export does draw them.

## 6. Follow The Tree

The fitted model answers questions about the tree directly:

```python
a, b = 0, 100
print(model.path(a, b))       # molecules along the tree path from a to b
print(model.hops(a, b))       # number of edges on that path
print(model.distance(a, b))   # summed Jaccard distance along the path

steps = model.hops_from(a)    # edges from molecule a to every other molecule
```

`model.distances_from(a)` gives the summed distances instead of edge counts. Both make good color layers: `viz.add_color_layout("Hops from 0", steps.tolist())`.

## 7. Save A Shareable HTML File

```python
path = viz.write_html("cluster_65053.html")
print(path)
```

This writes one HTML file with the data embedded, which you can open in a browser without a server. The page loads a few JavaScript libraries from a CDN, so it needs internet access.

## 8. Serve A Larger Map Locally

```python
viz.serve(port=8050)
```

Use `serve()` when the map is large enough that one HTML file becomes inconvenient. It writes the page and its data files to a temporary folder, opens your browser, and keeps serving until you press Ctrl+C. To keep the files, or to put them on a web server, use `viz.write_static("cluster_65053_site/")` instead.

## 9. Save The Model Too

```python
model.save("cluster_65053.tmap")
loaded = TMAP.load("cluster_65053.tmap")
```

This is useful when you want to keep the fitted tree and coordinates for later analysis.

## 10. Script Version

If you want the same workflow as a script, run:

```bash
python examples/chemistry/molecules_tmap.py --nrows 3000 --output examples/cluster_65053.html
```

Add `--serve` if you want it to start a local viewer.

For a longer version with the full 64,000 molecules and more color layers, see [notebook 02](../notebooks/02_cheminformatics.ipynb).
