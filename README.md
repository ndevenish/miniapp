# Dials Spotfinding Miniapps

This repository contains separate implementations of the DIALS spotfinding
intended for optimisation attempts and comparisons. Different parts of the 
repository may work with different requirements/toolkits.

## Repository Structure

The simplest place to start is the [`baseline/`] folder. This contains a copy
of the spotfinding code in DIALS, and a separate standalone DIALS-free
implementation for easy comparison by other implementations.

| Folder Name   | Implementation                                             |
| ------------- | ---------------------------------------------------------- |
| [`h5read/`]   | A small C/C++ library to read hdf5 files in a standard way |
| `common/`     | Common utility code, like coloring output, or image comparison, is stored here. |
| [`baseline/`] | The standard Dials "Dispersion" spotfinder. This includes a standalone implementation that can be used by other miniapps for comparison. |
| `cuda/`       | WIP Development of CUDA implementation of spotfinding.     |
| `dpcpp/`      | WIP experiments on using dpcpp to implement on FPGAs.      |

[`h5read/`]: h5read/
[`baseline/`]: baseline/