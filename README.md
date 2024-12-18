pyParareal
============

Python Parareal code to analyse Parareal's wave propagation characteristics.

Attribution
-----------
You can freely use and reuse this code in line with the BSD license. 
If you use it (or parts of it) for a publication, please cite

@Article{Ruprecht2018,  
author="Ruprecht, Daniel",  
title="Wave propagation characteristics of Parareal",  
journal="Computing and Visualization in Science",  
year="2018",  
doi="10.1007/s00791-018-0296-z",  
url="https://doi.org/10.1007/s00791-018-0296-z"  
}

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1012274.svg)](https://doi.org/10.5281/zenodo.1012274)

How can I reproduce Figures from the publication?
-----------------

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

## Acknowledgements

This project has received funding from the [European High-Performance
Computing Joint Undertaking](https://eurohpc-ju.europa.eu/) (JU) under
grant agreement No 955701 ([TIME-X](https://www.time-x-eurohpc.eu/)).
The JU receives support from the European Union's Horizon 2020 research
and innovation programme and Belgium, France, Germany, and Switzerland.
This project also received funding from the [German Federal Ministry of
Education and Research](https://www.bmbf.de/bmbf/en/home/home_node.html)
(BMBF) grants  16HPC048.


Who do I talk to?
-----------------

This code is written by [Daniel Ruprecht](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht).
