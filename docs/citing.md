# Citing SonoSK

If SonoSK is used in published work, please cite the software:

Kooklan S. SonoSK: medical ultrasound simulation in Python. Zenodo, 2026.
[doi:10.5281/zenodo.23517377](https://doi.org/10.5281/zenodo.23517377)

```
@software{kooklan_sonosk,
  author    = {Kooklan, Soheil},
  title     = {{SonoSK}: medical ultrasound simulation in {Python}},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23517377},
  url       = {https://doi.org/10.5281/zenodo.23517377}
}
```

The DOI above always resolves to the latest release. Every release also
has its own DOI, so that others can reproduce results with the same code:

| Version | DOI |
|---------|-----|
| 0.1.0 | [10.5281/zenodo.23517378](https://doi.org/10.5281/zenodo.23517378) |

The "Cite this repository" button on the
[GitHub page](https://github.com/sonosk/sonosk) gives the citation in APA
and BibTeX form.

In the methods section, please give the SonoSK version
(`sonosk.__version__`), the transducer and medium parameters, the grid
spacing and, for imaging, the scatterer density and the transmit sequence.
A script that reproduces the figures is the best record of all of these.
