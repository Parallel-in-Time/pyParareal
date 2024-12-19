pyParareal
============

Python Parareal is a simple code that is mostly useful to produce matrix formulations of Parareal for linear initial value problems that can then be used for theoretical analysis. The code is not parallelized, it is a tool for theory, not performance measurements. 

Attribution
-----------
You can freely use and reuse this code in line with its [license](https://github.com/Parallel-in-Time/pyParareal/blob/4d4e59aa1efcf62b0eca206e20517ebd67b5afc9/LICENSE).
If you use it (or parts of it) for a publication, please cite

@article{Ruprecht2018,  
  author = {Ruprecht, Daniel},  
  doi = {10.1007/s00791-018-0296-z},  
  journal = {Computing and Visualization in Science},  
  pages = {1--17},  
  title = {Wave propagation characteristics of Parareal},  
  url = {https://doi.org/10.1007/s00791-018-0296-z},  
  volume = {19},  
  year = {2018}  
}

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1012274.svg)](https://doi.org/10.5281/zenodo.1012274)

Structure of the code
-----------------
The main functionality is found in the classes located in the 

> ./src

folder. Scripts to produce various figures can be found in the

> ./scripts

folder and its subfolders. Tests are located in

> ./tests

and can be run by typing

> pytest ./tests/

while in the base folder of the code.

Dependencies
-----------------

The file 

> environment.yml

specifies the used [Anaconda](https://www.anaconda.com/) environment. If Anaconda is installed, the [environment can be created](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) by typing

> conda env create -f environment.yml

However, to use all functionality, you will also need a working installation of the [Dedalus software](https://dedalus-project.readthedocs.io/en/latest/pages/installation.html).

How can I reproduce Figures from the paper "Wave propagation characteristics of Parareal"?
-----------------

The scripts to reproduce the figures from this paper can be found [here](scripts/computing_and_visualization_in_science_2018).

 - Fig. 1 and Fig. 2 --> scripts/plot_svd_vs_dt.py
 - Fig. 3 --> scripts/plot_dispersion.py with parameter nu=0.0 and nu=0.1
 - Fig. 4 --> example/run.py
 - Fig. 5 --> scripts/plot_ieuler_dispersion.py
 - Fig. 6 --> scripts/plot_dispersion.py with stencil=2 (also set nu=0.0 and dx=1.0)
 - Fig. 7 --> scripts/plot_dispersion.py with artificial_coarse=1 and artifical_coarse=2
 - Fig. 8 --> example/run.py with artificial_coarse=2
 - Fig. 9 --> scripts/plot_svd_vs_waveno_different_G.py
 - Fig. 10 -> scripts/plot_dispersion.py with artifical_fine=1
 - Fig. 11 -> example/run.py with artifical_fine=1
 - Fig. 12 -> scripts/plot_dispersion.py with ncoarse=2
 - Fig. 13 -> scripts/plot_dispersion.py with Tend=64 and Nsamples=120
 - Fig. 14 -> scripts/plot_svd_vs_P.py
 - Fig. 15 -> scripts/plot_conv_vs_waveno.py
 - Fig. 16 -> scripts/plot_svd_vs_waveno.py

Pseudo-spectrum of the Parareal iteration matrix
-----------------
The code now also offers the possibility to compute the [pseudo-spectrum and pseudo-spectral radius](https://doi.org/10.1007/978-3-662-03972-4_6) of the Parareal iteration matrix.
You can also uses these parts in line with the in line with its [license](https://github.com/Parallel-in-Time/pyParareal/blob/4d4e59aa1efcf62b0eca206e20517ebd67b5afc9/LICENSE).
If you do, please also cite the following paper in addition to the one stated above:

tba

How can I reproduce Figures from the paper "Impact of spatial coarsening on Parareal convergence"?
-----------------

The scripts to reproduce the figures from this paper can be found [here](scripts/pseudo-spectrum).

 - figure_1-4.py with command line parameter 1, 2, 3 or 4 produces files figure_1.pdf to figure_4.pdf
 - figure_5-8.py with command line parameter 5, 6, 7 or 8 produces files figure_5.pdf to figure_8.pdf
 - figure_9-10.py with command line parameter 9 or 10 produces files figure_9.pdf and figure_10.pdf
 - figure_11-12.py with command line parameter 11 or 12 produces files figure_11.pdf and figure_12.pdf
 - figure_13-14.py with command line parameter 13 or 14 produces files figure_13.pdf and figure_14.pdf
 - figure_15-16.py with command line parameter 15 or 16 produces files figure_15.pdf and figure_16.pdf

![image](https://github.com/user-attachments/assets/9a0c18d7-1238-4e05-9d50-5640a61a9fd5)
   
## Acknowledgements

This project has received funding from the [European High-Performance
Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU) under
grant agreement No 955701 ([TIME-X](https://www.time-x-eurohpc.eu/)).
The JU receives support from the European Union's Horizon 2020 research
and innovation programme and Belgium, France, Germany, and Switzerland.
This project also received funding from the [German Federal Ministry of
Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html)
(BMBF) grants  16HPC048.

<p align="center">
  <img src="./img/EuroHPC.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./img/LogoTime-X.png" height="105" /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./img/BMBF_gefoerdert_2017_en.jpg" height="105" />
</p>

Who do I talk to?
-----------------

This code is written by [Daniel Ruprecht](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht).
