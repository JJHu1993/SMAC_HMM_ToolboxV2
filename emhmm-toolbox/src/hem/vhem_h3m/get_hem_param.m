function [mopt, emopt] = get_hem_param(N,M,K,Nv,tau,...
    init,initoptiter,initopttrials,termmode,termvalue,...
    max_iter,min_iter,reg_cov,trials,...
    inf_norm,smooth) 
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

mopt.N = N;
mopt.M = M;
mopt.emit.type= 'gmm';
mopt.emit.covar_type= 'diag';
mopt.K = K;
mopt.tau = tau;
mopt.Nv = Nv;

% mopt.initmode = 'r'; % random
% mopt.initmode = 'p'; % from some part
% mopt.initmode = 'g'; % from gaussian

mopt.initmode = init;
mopt.initopt.iter = initoptiter;
mopt.initopt.trials = initopttrials;



   
% rule for termination of EM iterations
mopt.termmode = termmode;
mopt.termvalue = termvalue;
% max number of iterations
mopt.max_iter = max_iter;
% max number of time series to use for learning
mopt.min_iter = min_iter;

emopt.trials = trials;




mopt.reg_cov = reg_cov;




if ~exist('inf_norm','var') || isempty(inf_norm)
    inf_norm = '';
end

if ~exist('smooth','var') || isempty(smooth)
    smooth = 1;
end

mopt.inf_norm = inf_norm;
mopt.smooth = smooth;
