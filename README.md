# HiD-PTM

**Hierarchical Disease-conditioned PTM Association Model**

HiD-PTM is a deep learning model for predicting associations between post-translational modification (PTM) sites and diseases. It jointly models five PTM types: **Acetylation, Methylation, Phosphorylation, SUMOylation, and Ubiquitination**.

## Model Overview

The model integrates local site features, global protein context, and disease representations:

- **Site and protein encoding:** frozen ESM-2 embeddings represent a 31-residue window centered on each PTM site and the mean representation of its full-length protein. PTM-type and positional embeddings provide additional context.
- **Disease encoding:** BiomedBERT semantic features and a two-layer GCN over the MONDO disease ontology form disease representations.
- **Disease-conditioned cross-attention:** the disease representation guides attention over residues in the site window.
- **Hierarchical interaction:** Protein–Disease and Site–Protein–Disease gates select and incorporate disease-relevant protein context into the site representation.
- **Association prediction:** an interaction gate and an MLP produce a site–disease association score.

## Repository Contents

This repository provides the model implementation, dataset and split information, five-fold trained checkpoints, and cross-validation results.

- [model.py](model.py): HiD-PTM implementation.
- [models/](models/): best checkpoints from five-fold cross-validation.
- [data/](data/): PTM site–disease dataset and fixed splits.
- [results/](results/): training records, predictions, and evaluation results.
- [source_code/](source_code/): source training and experiment scripts.
- [metadata/](metadata/): dataset and feature provenance.

Large data files and checkpoints are stored using **Git LFS**. The released checkpoints are five-fold models; the full precomputed feature cache is not included.
