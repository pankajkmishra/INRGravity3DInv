
# INRGravity3DInv

Python experiments for 3D gravity inversion with implicit neural representations (INRs).

The scripts in this repository test positional encoding, model capacity, discretization effects, explicit versus implicit regularization, ensemble variability, noise sensitivity, and related inversion settings on synthetic block-model examples.

## Requirements

- Python 3.11
- Dependencies in `requirements.txt`

## Exact Reproduction

The current `001-EncodingComparisons_blocky.py` result has been tested with the following runtime:

- OS: Windows 10
- Python: 3.11.14
- NumPy: 1.26.4
- Matplotlib: 3.10.7
- PyTorch: 2.9.1+cpu
- Device: CPU (`torch.cuda.is_available() == False`)

To reproduce that exact stack with Conda:

```bash
conda create -n sciml-repro python=3.11.14 pip -y
conda activate sciml-repro
pip install -r requirements.txt
python .\001-EncodingComparisons_blocky.py
```

To reproduce the examples exactly, please consider using these versions.

The script now prints its runtime information at startup. If a user reports a different figure, compare the printed Python, NumPy, Matplotlib, PyTorch, and device values first.

Expected terminal summary for the validated CPU run:

```text
Positional Encoding RMS density-contrast error ≈ 64.20 kg/m^3
Positional Encoding RMS data misfit ≈ 0.004 mGal | best epoch = 499 | best loss = 2.046
Hash Encoding RMS density-contrast error ≈ 64.22 kg/m^3
Hash Encoding RMS data misfit ≈ 0.003 mGal | best epoch = 233 | best loss = 0.999
```

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



