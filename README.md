pyParareal {#mainpage}
============

Python Parareal code to analyse Parareal's wave propagation characteristics.

Attribution
-----------
You can freely use and reuse this code in line with the BSD license. 
If you use it (or parts of it) for a publication, please cite

@unpublished{Ruprecht2017,   
  author = {Ruprecht, Daniel},    
  title = {{Wave propagation characteristics of Parareal}},    
  year = {2017}    
}

(this will be updated once published).

How can I reproduce Figures from the publication?
-----------------

 - Fig. 1 --> scripts/plot_dispersion.py with different parameters
 - Fig. 2 --> example/run.py
 - Fig. 3 --> scripts/plot_ieuler_dispersion.py
 - Fig. 4 --> scripts/plot_dispersion.py with approx_symbol=True
 - Fig. 5 --> scripts/plot_dispersion.py with artificial_coarse=1 or 2
 - Fig. 6 --> example/run.py with artificial_coarse=2
 - Fig. 7 --> scripts/plot_svd_vs_waveno_different_G.py
 - Fig. 8 --> scripts/plot_dispersion.py with artifical_fine=1
 - Fig. 9 --> example/run.py with artifical_fine=1
 - Fig. 10 -> scripts/plot_dispersion.py with ncoarse=2
 - Fig. 11 -> scripts/plot_dispersion.py with Tend=64
 - Fig. 12 -> scripts/plot_svd_vs_P.py
 - Fig. 13 -> scripts/plot_conv_vs_waveno.py
 - Fig. 14 -> scripts/plot_svd_vs_waveno.py 


Who do I talk to?
-----------------

This code is written by [Daniel Ruprecht](http://www.parallelintime.org/groups/leeds.html).