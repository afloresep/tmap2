# Visualization Guide

`TmapViz` is the public visualization object in TMAP.

The most useful mental model is simple:

- `show(...)` for notebooks
- `write_html(...)` for one shareable file
- `serve(...)` or `write_static(...)` for larger maps
- `model.plot_static(...)` for figures

This guide uses the molecule example from `examples/cluster_65053.csv`. Run the code from the repository root. In a script, put it under `if __name__ == "__main__":`, because the chemistry helpers start worker processes.

## Build A Base Visualization

```python
import pandas as pd
from tmap import TMAP
from tmap.utils import fingerprints_from_smiles, molecular_properties, murcko_scaffolds

df = pd.read_csv("examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

fps, valid = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048, return_valid=True)
smiles = [smi for smi, ok in zip(smiles, valid) if ok]

props = molecular_properties(smiles, properties=["mw", "logp", "n_rings", "qed", "tpsa", "hbd"])
scaffolds = murcko_scaffolds(smiles)

model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(fps)

viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
viz.add_color_layout("Ring Count", props["n_rings"].tolist(), categorical=True, color="tab10")
viz.add_color_layout("QED", props["qed"].tolist(), color="magma")
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

`return_valid=True` drops the SMILES RDKit can't parse. Filtering `smiles` with it keeps the properties, scaffolds and tooltips on the right molecules.

## Notebook Workflow

Use notebook widgets when you want fast iteration in Jupyter. They need the `notebook` extra (`pip install "tmap2[notebook]"`).

```python
viz.show(width=1000, height=650, controls=True)
```

`viz.to_widget(...)` builds the same jupyter-scatter object without displaying it, if you want to customize it first.

Good for:

- switching color layouts
- filtering categories
- inspecting tooltips
- staying inside a notebook

Current limitations:

- The widget draws points only. It warns about the tree edges and skips them; the HTML export draws them. Use `model.to_tmapviz(include_edges=False)` if you only need the widget and want to avoid the warning.
- The widget's filter menu lists categorical columns only. Continuous `add_filter` columns and the `filterable` setting apply to the HTML output.

For more on widgets, see [notebook 08](../notebooks/08_jscatter_demo.ipynb).

## HTML Workflow

Use HTML when you want one file you can send to someone else.

```python
path = viz.write_html("cluster_65053.html")
print(path)
```

Good for:

- sharing a result
- opening the map in a browser without a server
- keeping edge rendering

The data is embedded in the file. The page loads a few JavaScript libraries from a CDN, so the person opening it needs internet access.

## Larger Maps

For maps with millions of points, one HTML file gets heavy. Two options keep the data in separate files that the page fetches over HTTP, loading color columns only when you select them:

```python
viz.write_static("cluster_65053_site/")   # files for any static web host
viz.serve(port=8050)                      # same files in a temp folder, served locally
```

`serve()` opens your browser at `http://127.0.0.1:8050` (pass `open_browser=False` to skip that) and keeps running until you press Ctrl+C. In Jupyter that blocks the kernel, so use `write_static()` there and serve the folder from a terminal, for example with `python -m http.server`.

## Adding Data Columns

### Numeric Properties

```python
viz.add_color_layout("MW", props["mw"].tolist(), color="viridis")
viz.add_color_layout("LogP", props["logp"].tolist(), color="plasma")
```

Use this for values you want to color continuously. Continuous layouts take a gradient colormap such as `viridis` (the default), `plasma`, `magma` or `coolwarm`.

### Categorical Properties

```python
viz.add_color_layout(
    "Ring Count",
    props["n_rings"].tolist(),
    categorical=True,
    color="tab10",
)
```

Use this for discrete labels or small integer sets. Categorical layouts take a qualitative palette such as `tab10` (the default), `tab20`, `Set1` or `Dark2`. You can also pass your own colors, either as a list of hex strings (one per category, in order of first appearance) or as a dict from category to hex color:

```python
viz.add_color_layout("Series", series, categorical=True, color={"A": "#e41a1c", "B": "#377eb8"})
```

A gradient colormap on a categorical layout, or a qualitative palette on a continuous one, raises an error.

### Filter-Only Columns

```python
viz.add_filter("TPSA", props["tpsa"].tolist())
viz.add_filter("H-bond Donors", props["hbd"].astype(int).tolist(), categorical=True)
```

Use this when people should be able to filter on a column, but it does not need
to be one of the colors they can switch between. Numbers get a slider, groups
get one clickable bar each. Choosing colors is the slow part of
`add_color_layout`, and this skips it, so extra filter columns cost very little.
A name can only be one or the other: calling `add_filter` with the name of an
existing color layout raises an error.

Color layouts are already filterable, so the panel shows the `add_filter`
columns and the color layouts together. To say exactly what the panel shows,
set `filterable`:

```python
viz.filterable = ["MW", "LogP", "TPSA"]
```

Pass `add_as_label=False` to keep a filter column out of the tooltip.

### Tooltip Labels

```python
viz.add_label("Murcko Scaffold", scaffolds.tolist())
```

Use labels for text that should appear in the tooltip but should not control colors. Label columns are also what the search box looks through. To limit search to some columns, set `searchable`:

```python
viz.searchable = ["Murcko Scaffold"]
```

### Molecular Structures

```python
viz.add_smiles(smiles)
```

This enables SMILES structure rendering in tooltips and cards. The column is called `"SMILES"` unless you pass `name=`.

### Images, Proteins And 3D Structures

| Method | What each value should be |
|--------|---------------------------|
| `viz.add_images(values)` | A `data:image/png;base64,...` string per point, shown as a thumbnail |
| `viz.add_protein_ids(values)` | A UniProt accession. Cards load the AlphaFold structure and UniProt details on demand |
| `viz.add_3d_structures(values, source="url")` | A URL or relative path to a PDB/mmCIF file, fetched when a card opens |
| `viz.add_3d_structure_files(paths)` | A local PDB/mmCIF path. `write_html` and `write_static` copy the files into a `structures/` folder next to the page |

[Notebook 13](../notebooks/13_local_protein_structures.ipynb) walks through the structure options.

## Cards, Links And Formatting

Clicking a point pins a card with its details. `configure_card` picks what goes on it:

```python
viz.configure_card(
    title_column="Murcko Scaffold",
    fields=["MW", "LogP", "QED", "SMILES"],
    links=[{"label": "PubChem", "url": "https://pubchem.ncbi.nlm.nih.gov/#query={SMILES}"}],
)
```

In link URLs, `{column name}` is replaced with that point's value in the column. Values are URL-encoded, so build the link from an ID column; a column that already holds full URLs won't work as a link template.

`configure_column` sets per-column display hints:

```python
viz.configure_column("MW", display_name="Molecular weight", format="fixed:1")
viz.configure_column("QED", format="fixed:2", copyable=True)
```

`format` accepts `"fixed:N"`, `"percent:N"`, `"scientific:N"`, `"integer"` and `"stars:N"`. See [notebook 09](../notebooks/09_card_configuration.ipynb) for more card layouts.

## Styling

Page settings are plain attributes on the `TmapViz`:

```python
viz.background_color = "#FFFFFF"
viz.point_size = 3.0
viz.opacity = 0.8
viz.set_edge_style(color="#999999", width=1.0, opacity=0.4)
```

## Static Figures

For papers and slides, draw the map with matplotlib:

```python
ax = model.plot_static(color_by=props["qed"], color_map="magma", point_size=2)
ax.figure.savefig("cluster_65053.png", dpi=300)
```

Tree edges are drawn behind the points by default (`edges=False` turns them off). For an embedding that didn't come from a fitted model, use `tmap.visualization.plot_static(embedding, edges=tree_edges, ...)`.

## Shortcuts On The Model

If you don't need extra columns, the fitted model can skip the `TmapViz` step:

| Call | Same as |
|------|---------|
| `model.to_html("map.html", title="My map")` | `to_tmapviz()`, set the title, `write_html()` |
| `model.serve(port=8050)` | `to_tmapviz().serve()` |
| `model.plot(color_by=..., data=df)` | A jupyter-scatter widget colored by a column of `df` |

## Common Patterns

### Quick Notebook First, HTML Second

```python
viz.show(width=1000, height=650, controls=True)

viz.write_html("cluster_65053.html")
```

### Save Model And Visualization

```python
model.save("cluster_65053.tmap")
viz.write_html("cluster_65053.html")
```

`TMAP.load("cluster_65053.tmap")` brings the model back. `TmapViz` objects are not saved; rebuild them from the loaded model.

### Get The Data Back

```python
df_viz = viz.to_dataframe()
```

This returns every column plus the map coordinates as a pandas DataFrame, which is handy for working with a lasso selection in a notebook.

### Start Small

For tutorials, start with `nrows=1000` or `nrows=3000`. Once the workflow is working, use more rows.

For a step-by-step chemistry example, see the [molecule tutorial](molecule_tutorial.md).
