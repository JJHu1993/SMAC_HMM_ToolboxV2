========================================================================

  Eye-Movement analysis with Hidden Markov Models (HMMs)
  emhmm-toolbox

  Antoni B. Chan, City University of Hong Kong
  Janet H. Hsiao, University of Hong Kong
  Tim Chuk, University of Hong Kong

  Copyright (c) 2017, City University of Hong Kong & University of Hong Kong

========================================================================

--- DESCRIPTION ---
This is a MATLAB toolbox for analyzing eye movement data with hidden Markov Models (HMMs). 
The major functions of the toolbox are:
  1) Estimating HMMs for an individual's eye-gaze data.
  2) Clustering individuals' HMMs to find common strategies.
  3) Visualizing HMMs
  4) Statistical tests to see if two HMMs are different.

--- BRIEF INSTRUCTIONS ---
In MATLAB, run "setup" to setup the paths.
In demo folder, run "demo_faces" for an example.
Also see "demo_faces_jov_clustering" and "demo_faces_jov_compare".

More documentation and descriptions are in the "docs" folder.


--- REFERENCES ---
If you use this toolbox, please cite the following papers:

For learning HMMs for eye gaze data:
  Tim Chuk, Antoni B. Chan, and Janet H. Hsiao.
  "Understanding eye movements in face recognition using hidden Markov models."
  Journal of Vision, 14(11):8, Sep 2014.

For clustering HMMs with the VHEM algorithm:  
  Emanuele Coviello, Antoni B. Chan, and Gert R.G. Lanckriet.
  "Clustering hidden Markov models with variational HEM".
  Journal of Machine Learning Research (JMLR), 15(2):697-747, Feb 2014.


--- CHANGE LOG ---
see emhmm_version.m for the version information.


--- CONTACT INFO ---
Please send comments, bug reports, feature requests to Antoni Chan (abchan at cityu dot edu . hk).

