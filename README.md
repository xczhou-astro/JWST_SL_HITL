# JWST Strong-Lens Search

This repository contains the code and catalogue associated with our search for galaxy-scale strong-lens candidates in JWST imaging. The workflow combines self-supervised representation learning, human-in-the-loop (HITL) candidate selection, nearest-neighbour retrieval, and expert visual inspection.

## Repository structure

### `HITL/`

The `HITL` directory contains the web application used for human-in-the-loop candidate selection. During each iteration, the classifier ranks sources using their learned representations, and the website presents selected images for human labelling. The accumulated positive and negative labels are then used to update the classifier for the next round.

### `BYOL/`

The `BYOL` directory contains the Bootstrap Your Own Latent (BYOL) implementation used for self-supervised representation learning. It learns morphological representations from galaxy images without requiring class labels. These representations are subsequently used for nearest-neighbour retrieval of sources resembling the human-selected strong-lens candidates.

### `VI/`

The `VI` directory contains the program used for the final visual inspection. Candidate systems are independently assessed by strong-lensing experts, and their grades are combined into a visual-inspection score.

## Candidate catalogue

The final visually inspected candidate catalogue is provided in [`strong_lens_candidate_catalogue.csv`](./strong_lens_candidate_catalogue.csv). It contains 1,697 inspected sources. These entries are strong-lens candidates rather than spectroscopically or lens-model confirmed systems.

The catalogue contains the following columns:

| Column | Description |
| --- | --- |
| `name` | Unique source identifier. |
| `grades` | Grades assigned independently by the eight visual inspectors. Each character records the grade from one inspector. |
| `score` | Combined visual-inspection score. Grades A, B/S, and U/X contribute 2, 1, and 0 points, respectively. The maximum possible score is 16. |
| `ra` | Right ascension in decimal degrees (ICRS). |
| `dec` | Declination in decimal degrees (ICRS). |
| `ABmag_F444W` | Source AB magnitude in the JWST/NIRCam F444W band. |
| `has_COWLS_counterpart` | Whether a positional counterpart is found in the COSMOS-Web Lens Survey (COWLS) catalogue. |
| `within_COWLS_footprint` | Whether the source lies within the COWLS survey footprint. |
| `code_COWLS` | Identifier or classification code of the matched COWLS source; blank when no counterpart is found. |

In this work, sources with `score > 4` form the higher-scoring candidate sample. Of these 53 candidates, 23 have COWLS counterparts, 15 lie within the COWLS footprint but have no positional counterpart, and 15 are located outside the COWLS footprint.

## Citation

If you use the code or catalogue, please cite the accompanying paper. Citation information will be added following publication.
