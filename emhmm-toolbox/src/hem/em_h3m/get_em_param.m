function [mopt, emopt] = get_em_param(N,M,K,init,termmode,termvalue,...
    max_iter,min_iter,max_obs,reg_cov,trials)
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

mopt.N = N;
mopt.M = M;
mopt.emit.type= 'gmm';
mopt.emit.covar_type= 'diag';
mopt.K = K;


mopt.initmode = init; 
% rule for termination of EM iterations
mopt.termmode = termmode;
mopt.termvalue = termvalue;
% max number of iterations
mopt.max_iter = max_iter;
% max number of time series to use for learning
mopt.max_obs = max_obs;
mopt.min_iter = min_iter;

emopt.trials = trials;


mopt.reg_cov = reg_cov;

% mopt.initmode =
% 'r'    random initialization of all parameters (this will probably never work well for you)
% 'p'    randomly partition the input time series in K group, and estimate each HMM component on one of the partition
% 'g'    first estimate a GMM on all the data, than initialize each HMM by setting emission to the GMM (with randomized weights) and using random parameters for the HMM dynamics
% 'gm'   similar to 'g', but uses MATLAB own function 
% 'km'   similar to 'g' but uses k means
% 'gL2R' similar to 'g'. but initialize HMMs as left to righ HMMs



