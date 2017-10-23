function [group_hmms] = vhem_cluster(hmms, K, S, hemopt)
% vhem_cluster - cluster hmms into groups using VHEM algorithm
%
% [group_hmms] = vhem_cluster(hmms, K, S, hemopt)
%
% This function uses a modified version of the H3M toolbox.
%
% INPUTS
%      hmms = Nx1 cell array of HMMs, learned with vbhmm_learn.
%
%         K = number of groups to use (required)
%         S = number of states (ROIs) in a group HMM 
%             (default=[], use the median number of states in input HMMs)
%
%    hemopt = structure containing other options as below:
%      hemopt.initmode = 'base'  - randomly select base HMMs components as initialization 
%                        'baseem' - randomly select base HMM emissions as initialization [default]
%                        'split' - use splitting strategy 
%      hemopt.trials  = number of trials with random initialization to run (default=50)
%      hemopt.reg_cov = regularization for covariance matrix (default=0.001)
%      hemopt.max_iter = maximum number of iterations
%      hemopt.sortclusters = '' (no sorting, default)
%                          = 'p' - sort ROIs by prior frequency
%                          = 'f' - sort ROIs by most likely fixation path [default]
%                            (see vbhmm_standardize for more options)
%      hemopt.verbose  = 0 - no messages
%                        1 - progress messages [default]
%                        2 - more messages
%                        3 - debugging
%
% OUTPUT
%   group_hmm = structure containing the group HMMs and information
%   group_hmm.hmms       = cell array of group HMMs {1xK}
%   group_hmm.label      = the cluster assignment for each input HMM [1xN]
%   group_hmm.groups     = indices of input HMMs in each group {1xK}
%   group_hmm.group_size = size of each group [1xK]
%   group_hmm.Z          = probabilities of each input HMM belonging to a group [Nx2]
%   group_hmm.LogL       = log-likelihood score
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2016-11-21: ABC - initial version (modified from Tim)
% 2016-12-09: ABC - added initialization using splitting
% 2016-01-10: ABC - added initialization using base emissions (baseem)
% 2017-01-11: ABC - added verbose option
% 2017-01-19: ABC - add fixation duration

% TODO:
%  - option for 'full' covariance

if nargin<3
  S = [];
end
if nargin<4
  hemopt = struct;
end

% number of clusters
hemopt.K = K;
hemopt = setdefault(hemopt, 'verbose', 1);

VERBOSE_MODE = hemopt.verbose;

% number of states in a group HMM
if isempty(S)
  % get median number
  hmmsS = cellfun(@(x) length(x.prior), hmms);
  S = ceil(median(hmmsS)); % if in the middle, then take the larger S
  
  if (VERBOSE_MODE >= 1)
    fprintf('using median number of states: %d\n', S);
  end
end
hemopt.N = S;
%hemopt = setdefault(hemopt, 'N',  3); % Number of states of each HMM

% setable parameters (could be changed)
hemopt = setdefault(hemopt, 'trials', 50);     % number of trials to run
hemopt = setdefault(hemopt, 'reg_cov', 0.001);   % covariance regularizer
hemopt = setdefault(hemopt, 'termmode', 'L');   % rule to terminate EM iterations
hemopt = setdefault(hemopt, 'termvalue', 1e-5);
hemopt = setdefault(hemopt, 'max_iter', 100);    % max number of iterations
hemopt = setdefault(hemopt, 'min_iter', 1);     % min number of iterations
hemopt = setdefault(hemopt, 'sortclusters', 'f');
hemopt = setdefault(hemopt, 'initmode', 'baseem');  % initialization mode

% standard parameters (not usually changed)
hemopt = setdefault(hemopt, 'Nv', 100); % number of virtual samples
hemopt = setdefault(hemopt, 'tau', 10); % temporal length of virtual samples
hemopt = setdefault(hemopt, 'initopt', struct);   % options for 'gmm' initaliation (unused)
hemopt.initopt = setdefault(hemopt.initopt, 'iter', 30);  % number of GMM iterations for init (unused)
hemopt.initopt = setdefault(hemopt.initopt, 'trials', 4); % number of GMM trials for init (unused)
hemopt = setdefault(hemopt, 'inf_norm', 'nt');   % normalization before calculating Z ('nt'=tau*N/K). This makes the probabilites less peaky (1 or 0).
hemopt = setdefault(hemopt, 'smooth', 1);        % smoothing parameter - for expected log-likelihood

% fixed parameters (not changed)
hemopt.emit.type= 'gmm';
hemopt.emit.covar_type= 'diag';  % 'full' is not supported yet
hemopt.M = 1; % number of mixture components in each GMM for an emission (should always be 1)


% a separate structure
emopt.trials = hemopt.trials;

% convert list of HMMs into an H3M
H3M = hmms_to_h3m(hmms, 'diag');

% run VHEM clustering
h3m_out = hem_h3m_c(H3M, hemopt, emopt);

% convert back to our format
group_hmms = h3m_to_hmms(h3m_out);

% sort clusters
if ~isempty(hemopt.sortclusters)
  group_hmms = vbhmm_standardize(group_hmms, hemopt.sortclusters);
end
  
% save parameters
group_hmms.hemopt = hemopt;

function hemopt = setdefault(hemopt, field, value)
if ~isfield(hemopt, field)
  hemopt.(field) = value;
end


