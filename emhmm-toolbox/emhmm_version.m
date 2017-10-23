function v = emhmm_version()
% emhmm_version - display the toolbox version
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

%% HISTORY
% v0.0 2016/06    - added VB-HMM for learning individual's HMMs
% v0.0 2016/06/17 - distribution 20160617
% v0.1 2016/08    - initialize VB with ROIs, and use fixed ROIs
% v0.2 2016/11/21 - added HEM clustering, ttest for HMMs, rearranged source files, added demo code
% v0.3 2016/12/09 - added more demos, added component splitting for initialization (random is still better).
% v0.4 2016/01/13 - added xls reader and demo; use random initialization; added documentation (docs)
% v0.5 2016/01/21 - updated to handle fixation duration with xy; bug fixes; automatically select some hyperparameters.
%                 - updated for Antoine: options for gmdistribution.fit (random_gmm_opt); 

%% current version
v = 0.5;