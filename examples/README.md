# Examples

Runnable scripts grouped by data type. Run them from the repository root. Each
script's docstring lists the extra packages it needs (for example `torch`,
`datasets` or `sentence-transformers`). Datasets and cached embeddings go to
`examples/data/`; some raw datasets and model weights use their library's own
cache instead (Hugging Face, scikit-learn, torchvision). Outputs are written to
`examples/`.

```bash
python examples/chemistry/molecules_tmap.py --nrows 3000
```

## Chemistry

| Example | Description |
|---------|-------------|
| [`chemistry/molecules_tmap.py`](chemistry/molecules_tmap.py) | The simplest chemistry example: SMILES → fingerprints → interactive map, with molecular properties and Murcko scaffolds |
| [`chemistry/molecules_tmap_legacy.py`](chemistry/molecules_tmap_legacy.py) | `molecules_tmap.py` with the adaptive layout and untangle post-pass switched off, for comparison |

## Images

| Example | Description |
|---------|-------------|
| [`images/pet_breed_audit.py`](images/pet_breed_audit.py) | Audit an image classifier: ResNet-50 embeddings of Oxford-IIIT Pets, a linear probe, and tree analysis of where it fails |
| [`images/mnist_cosine_tmap.py`](images/mnist_cosine_tmap.py) | MNIST digits with the cosine metric, including paths between similar digits |
| [`images/emnist_characters_tmap.py`](images/emnist_characters_tmap.py) | Handwritten digits and letters together, showing where OCR confuses the two |
| [`images/flowers_tmap.py`](images/flowers_tmap.py) | Oxford Flowers 102: morphological gradients across species |
| [`images/cub200_birds_tmap.py`](images/cub200_birds_tmap.py) | CUB-200 birds: morphological paths across 200 species |
| [`images/wikiart_tmap.py`](images/wikiart_tmap.py) | WikiArt paintings colored by artistic style |

## Proteins

| Example | Description |
|---------|-------------|
| [`proteins/esm_atlas_tmap.py`](proteins/esm_atlas_tmap.py) | Two views of ESMC's protein space: raw embeddings and SAE features, with predicted structures in the pinned cards |

## Text

| Example | Description |
|---------|-------------|
| [`text/word_embeddings_tmap.py`](text/word_embeddings_tmap.py) | About 1,200 common English words in 19 categories, embedded with a sentence-transformer |
| [`text/word_embeddings_50k.py`](text/word_embeddings_50k.py) | The same idea at 50,000 WordNet nouns |

## Layout internals

| Example | Description |
|---------|-------------|
| [`layout/untangle_demo.py`](layout/untangle_demo.py) | Before-and-after figure showing what the crossing-reduction (untangle) post-pass does to a layout |

## Data

- `cluster_65053.csv`: about 64,000 SMILES from an Enamine chemical cluster. Used
  by the chemistry examples, the layout demo, and several docs and notebooks.
- `data/`: datasets and cached embeddings downloaded by the examples on first
  run. Not tracked in git, and can be deleted to reclaim the space.

Scripts write their outputs (`.html` maps, `.png` figures, `.txt` reports, and
`esm_atlas_out/` for the protein example) to `examples/`. Only the `.html` files
and `data/` are in `.gitignore`, so the others show up in `git status`.

## Notebooks

Step-by-step walkthroughs live in [`notebooks/`](../notebooks). The lower-level
MinHash → LSHForest → OGDF layout → `TmapViz` pipeline is in
[`notebooks/12_lower_level_pipeline.ipynb`](../notebooks/12_lower_level_pipeline.ipynb).
