Graph based privacy attack on multiple encoding techniques for PPRL
=======================================================================

Anushka Vidanage, Peter Christen, Thilina Ranbaduge, and Rainer Schnell

Paper title: A Graph Matching Attack on Privacy-Preserving Record Linkage


Copyright 2020 Australian National University and others.
All Rights reserved.

See the file COPYING for the terms under which the computer program
code and associated documentation in this package are licensed.

10 September 2020.

Contact: anushka.vidanage@anu.edu.au, peter.christen@anu.edu.au

-------------------------------------------------------------------

Requirements:
=============

The Python programs included in this package were written and
tested using Python 2.7.6 running on Ubuntu 16.04

The following extra packages are required:
- numpy
- scipy
- networkx
- pickle
- gensim.models
- matplotlib
- sklearn

Running the attack program:
===========================

To run the program, use the following command (with an example setting):

  python graph-attack-pprl.py euro-census.csv 0 , True [1,2,8] -1 
         euro-census.csv 0 , True [1,2,8] -1 2 False dice True bf rh 
         15 1000 clk none None dice

For moe details about the command line arguments see comments at the top of 
'graph-attack-pprl.py'
