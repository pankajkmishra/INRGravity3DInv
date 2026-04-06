
# INRGravity3DInv

Python experiments for 3D gravity inversion with implicit neural representations (INRs).

The scripts in this repository test positional encoding, model capacity, discretization effects, explicit versus implicit regularization, ensemble variability, noise sensitivity, and related inversion settings on synthetic block-model examples.

## Requirements

- Python
- `numpy`
- `matplotlib`
- `torch`

## Run

From the repository folder, run any experiment directly. Main scripts include:

- `001-EncodingComparison_smooth.py`
- `001-EncodingComparisons_blocky.py`
- `002a-BlockNetworkSizeComparison.py`
- `002b-BlockGridSizeComparison.py`
- `003-ImplictvsExplicitRegularisation.py`
- `004-ModelEnsambles.py`
- `005-NoiseSensitivity.py`

For example:

```bash
python .\001-EncodingComparisons_blocky.py
python .\002a-BlockNetworkSizeComparison.py
python .\002b-BlockGridSizeComparison.py
python .\003-ImplictvsExplicitRegularisation.py
python .\004-ModelEnsambles.py
python .\005-NoiseSensitivity.py
```

Each script creates figures and metrics in the `plots/` directory.

## Citation

Mishra, P.K., Laaksonen, S., Kamm, J. and Singh, A., 2025. Three-dimensional inversion of gravity data using implicit neural representations. arXiv preprint arXiv:2510.17876. Available at: https://doi.org/10.48550/arXiv.2510.17876.

```bibtex
@article{mishra2025inrgravity,
	title = {Three-dimensional inversion of gravity data using implicit neural representations},
	author = {Mishra, Pankaj K and Laaksonen, Sanni and Kamm, Jochen and Singh, Anand},
	journal = {arXiv preprint arXiv:2510.17876},
	year = {2025},
	doi = {10.48550/arXiv.2510.17876},
	url = {https://doi.org/10.48550/arXiv.2510.17876}
}
```



