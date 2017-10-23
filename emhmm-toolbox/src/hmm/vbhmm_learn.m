% vbhmm_learn - learn HMM with variational Bayesian EM algorithm
%
% [hmm,L] = vbhmm_learn(data,K,vbopt)
%
% INPUTS
%   data = Nx1 cell array, each element is a fixation sequence: 
%             data{i} = [TxD] matrix, where each row is a fixation (x,y) or (x,y,t)
%          N = number of sequences, T = sequence length (can be different for each i)
%          D = dimension: 2 for fixation location (x,y); 3 for location & duration (x,y,d). Duration should be in milliseconds.
%          
%      K = scalar: number of hidden states (clusters)
%          vector: automatically selects the number of hidden states from the given values. 
%                  The model K with highest log-likelihood is selected.
%
%  vbopt = structure containing other options as below:
%
%   VB hyper-parameters:
%     vbopt.alpha = Dirichlet distribution concentration parameter -- large value
%                   encourages uniform prior, small value encourages concentrated prior (default=0.1)
%                   Another way to think of it is in terms of "virtual" samples -- A typical way to 
%                   estimate the probability of something is to count the number of samples that it
%                   occurs and then divide by the total number of samples, 
%                   i.e. # times it occurred / # samples.  
%                   The alpha parameter of the Dirichlet adds a virtual sample to this estimate, 
%                   so that the probability estimate is (# times it occurred + alpha) / # samples.  
%                   So for small alpha, then the model will basically just do what the data says.  
%                   For large alpha, it forces all the probabilities to be very similar (i.e., uniform).
%     vbopt.mu    = prior mean - it should be dimension D;
%                   for D=2 (fixation location), default=[256;192] -- typically it should be at the center of the image.
%                   for D=3 (location & duration), default=[256,192,250]
%     vbopt.W     = size of the inverse Wishart distribution (default=0.005).
%                   If a scalar, W is assumed to be isotropic: W = vbopt.W*I.
%                   If a vector, then W is a diagonal matrix:  W = diag(vbopt.W);
%                   This determines the covariance of the ROI prior.
%                   For example W=0.005 --> covariance of ROI is 200 --> ROI has standard devation of 14.
%                   
%     vbopt.v     = dof of inverse Wishart, v > D-1. (default=10) -- larger values give preference to diagonal covariance matrices.
%     vbopt.beta  = Wishart concentration parameter -- large value encourages use of
%                   prior mean & W, while small value ignores them. (default=1)
%     vbopt.epsilon = Dirichlet distribution for rows of the transition matrix (default=0.1). 
%                     The meaning is similar to alpha, but for the transition probabilities.
%  
%   EM Algorithm parameters
%     vbopt.initmode     = initialization method (default='random')
%                            'random' - initialize emissions using GMM with random initialization (see vbopt.numtrials)
%                            'initgmm' - specify a GMM for the emissions (see vbopt.initgmm)
%                            'split' - initialize emissions using GMM estimated with component-splitting
%     vbopt.numtrials    = number of trails for 'random' initialization (default=50)
%     vbopt.random_gmm_opt = for 'random' initmode, cell array of options for running "gmdistribution.fit".
%                            The cell array should contain pairs of the option name and value, which are recognized
%                            by "gmdistribution.fit".
%                            For example, {'CovType','diagonal','SharedCov',true,'Regularize', 0.0001}.
%                            This option is helpful if the data is ill-conditioned for the standard GMM to fit.
%                            The default is {}, which does not pass any options.
%     vbopt.initgmm      = initial GMM for 'initgmm':
%                            initgmm.mean{k} = [1 x K]
%                            initgmm.cov{k}  = [K x K]
%                            initgmm.prior   = [1 x K]                               
%     vbopt.maxIter      = max number of iterations (default=100)
%     vbopt.minDiff      = tolerence for convergence (default=1e-5)
%     vbopt.showplot     = show plots (default=1)
%     vbopt.sortclusters = '' - no sorting [default]
%                          'e' - sort ROIs by emission probabilites
%                          'p' - sort ROIs by prior probability
%                          'f' - sort ROIs by most-likely fixation path [default]
%                              (see vbhmm_standardize for more options)
%     vbopt.groups       = [N x 1] vector: each element is the group index for a sequence.
%                          each group learns a separate transition/prior, and all group share the same ROIs.
%                          default = [], which means no grouping used
%     cvopt.fix_cov      = fix the covariance matrices of the ROIs to the specified matrix.
%                          if specified, the covariance will not be estimated from the data.
%                          The default is [], which means learn the covariance matrices.
%     vbopt.fix_clusters = 1 - keep Gaussian clusters fixed (don't learn the Gaussians)
%                          0 - learn the Gaussians [default]
%     vbopt.
%
%     vbopt.verbose      = 0 - no messages
%                        = 1 - a few messages showing progress [default]
%                        = 2 - more messages
%                        = 3 - debugging
%
% OUTPUT
%   vb_hmm.prior       = prior probababilies [K x 1]
%   vb_hmm.trans       = transition matrix [K x K]: trans(i,j) is the P(x_t=j | x_{t-1} = i)
%   vb_hmm.pdf{j}.mean = ROI mean [1 x D]
%   vb_hmm.pdf{j}.cov  = covariances [D x D]
%   vb_hmm.LL          = log-likelihood of data
%   vb_hmm.gamma       = responsibilities gamma{i}=[KxT] -- probability of each fixation belonging to an ROI.
%   vb_hmm.M           = transition counts [K x K]
%   vb_hmm.N1          = prior counts [K x 1]
%   vb_hmm.N           = cluster sizes [K x 1] (emission counts)
%
%  for using groups
%   vb_hmm.prior{g}    = prior for group g
%   vb_hmm.trans{g}    = transition matrix for group g
%   vb_hmm.M{g}        = transition counts for group g [K x K]
%   vb_hmm.N1{g}       = prior counts for group g [K x 1]
%   vb_hmm.Ng{g}       = cluster sizes of group g
%   vb_hmm.N           = cluster sizes for all groups [K x 1] (emission counts)
%   vb_hmm.group_ids     = group ids
%   vb_hmm.group_inds{g} = indices of data sequences for group g
%   vb_hmm.group_map     = sanitized group IDs (1 to G)
%  
%  internal variational parameters - (internal variables)
%   vb_hmm.varpar.epsilon = accumulated epsilon
%   vb_hmm.varpar.alpha   = accumulated alpha
%   vb_hmm.varpar.beta    = accumulated beta
%   vb_hmm.varpar.v       = accumulated v
%   vb_hmm.varpar.m       = mean
%   vb_hmm.varpar.W       = Wishart
%
%  for model selection:
%   vb_hmm.model_LL    = log-likelihoods for all models tested
%   vb_hmm.model_k     = the K-values used
%   vb_hmm.model_bestK = the K with the best value.
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


% VERSIONS
% 2016-04-21: - initial version from Tim
% 2016-04-25: - speed optimization, documentation
% 2016-04-26: - fix initialization bugs (vbhmm_init): mix_t.W0 and mix_t.v
% 2016-04-27: - initialize with random trials
%             - expose parameters (minDiff, etc)
% 2016-04-29: - bug fix: calculate correct LL lower-bound (vbhmm_em)
%             - fix precision problem for FB algorithm when T is large (vbhmm_em)
%             - bug fix: xi was normalized by row for each sequence n
%                        it should be normalized by matrix for each time t
% 2016-05-05: - model selection over K
% 2016-05-06: - group data - learn separate transition/prior for each group, but share the same ROIs.
% 2016-08-14: - separate function for running FB algorithm (vbhmm_fb)
%             - save variational parameters for output
% 2016-08-15: - initialize with GMM (cell array)
%             - keep clusters fixed, just train the transition/prior parameters (fix_clusters option)
% 2016-12-09: - initialize using splitting-component GMM.
% 2017-01-11: - added verbose option
% 2017-01-17: - update for fixation duration
%             - bug fix: changed vbopt.mean to vbopt.mu
% 2017-01-21: - added for Antoine: constrain covariance to fixed matrix (fix_cov)
%             - added for Antoine: for initialization with random gmm, 
%                     added passable options for gmdistribution.fit (random_gmm_opt)

function [hmm,L] = vbhmm_learn(data,K,vbopt)

if nargin<3
  vbopt = struct;
end


vbopt = setdefault(vbopt, 'alpha', 0.1);  % the Dirichlet concentration parameter

if isfield(vbopt, 'mean')
  warning('DEPRECATED: vbopt.mean has been renamed to vbopt.mu');
  vbopt.mu = vbopt.mean;
end

D = size(data{1}, 2);
switch(D)
  case 2
    defmu = [256;192];
  case 3
    defmu = [256;192;150];
  otherwise
    error(sprintf('no default mu for D=%d', D));
end
vbopt = setdefault(vbopt, 'mu',   defmu); % hyper-parameter for the mean
vbopt = setdefault(vbopt, 'W',     .005); % the inverse of the variance of the dimensions
vbopt = setdefault(vbopt, 'beta',  1);
vbopt = setdefault(vbopt, 'v',     5);
vbopt = setdefault(vbopt, 'epsilon', 0.1);
  
vbopt = setdefault(vbopt, 'initmode',  'random');
vbopt = setdefault(vbopt, 'numtrials', 50);
vbopt = setdefault(vbopt, 'maxIter',   100); % maximum allowed iterations
vbopt = setdefault(vbopt, 'minDiff',   1e-5);
vbopt = setdefault(vbopt, 'showplot',  1);
vbopt = setdefault(vbopt, 'sortclusters', 'f');
vbopt = setdefault(vbopt, 'groups', []);
vbopt = setdefault(vbopt, 'fix_clusters', 0);

vbopt = setdefault(vbopt, 'random_gmm_opt', {});
vbopt = setdefault(vbopt, 'fix_cov', []);

vbopt = setdefault(vbopt, 'verbose', 1);

VERBOSE_MODE = vbopt.verbose;


%% run for multiple K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if length(K)>1
  % turn off plotting
  vbopt2 = vbopt;
  vbopt2.showplot = 0; 
  
  % call learning for each value of K
  out_all = cell(1,length(K));
  LLk_all  = zeros(1,length(K));
  for ki = 1:length(K)
    if (VERBOSE_MODE >= 2)
      fprintf('-- K=%d --\n', K(ki));
    elseif (VERBOSE_MODE == 1)
      fprintf('-- vbhmm K=%d: ', K(ki));
    end
    
    % for initgmm, select one
    if strcmp(vbopt2.initmode, 'initgmm')
      vbopt2.initgmm = vbopt.initgmm{ki};
    end
    
    % call learning with a single k
    out_all{ki} = vbhmm_learn(data, K(ki), vbopt2);
    LLk_all(ki) = out_all{ki}.LL;
  end

  % correct for multiple parameterizations
  LLk_all = LLk_all + gammaln(K+1);
  
  % get K with max data likelihood
  [maxLLk,ind] = max(LLk_all);
  
  % return the best model
  hmm       = out_all{ind};
  hmm.model_LL    = LLk_all;
  hmm.model_k     = K;
  hmm.model_bestK = K(ind);
  L = maxLLk;
  
  if VERBOSE_MODE >= 1
    fprintf('best model: K=%d; L=%g\n', K(ind), maxLLk);
  end
  
  if vbopt.showplot
    figure
    hold on
    plot(K, LLk_all, 'b.-')
    plot([min(K), max(K)], [maxLLk, maxLLk], 'k--');
    plot(K(ind), maxLLk, 'bo');
    hold off
    grid on
    xlabel('K');
    ylabel('data log likelihood');
  end


else 
  %% run for a single K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  switch(vbopt.initmode)
    %%% RANDOM initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    case 'random'
      numits  = vbopt.numtrials;
      vb_hmms = cell(1,numits);
      LLall   = zeros(1,numits);
      % run several iterations
      for it = 1:numits        
        if (VERBOSE_MODE == 1)
          fprintf('%d ', it);
        end   
        
        vb_hmms{it} = vbhmm_em(data, K, vbopt);
        LLall(it) = vb_hmms{it}.LL;
        
        % debugging
        %  vbhmm_plot(vb_hmms{it}, data);             
      end
      
      % choose the best
      [maxLL,maxind] = max(LLall);
      if (VERBOSE_MODE >= 1)        
        fprintf('\nbest run=%d; LL=%g\n', maxind, maxLL);
      end
      
      %LLall
      %maxLL
      %maxind
      hmm = vb_hmms{maxind};
      L = hmm.LL;
      
    %%% Initialize with learned GMM %%%%%%%%%%%%%%%%%%%%%%
    case 'initgmm'
      hmm = vbhmm_em(data, K, vbopt);
      L = hmm.LL;
    
    %%% Initialize with component-splitting GMM %%%%%%%%%%
    case 'split'
      hmm = vbhmm_em(data, K, vbopt);
      L = hmm.LL;
      
  end


end

% post processing
% - reorder based on cluster size
if ~isempty(vbopt.sortclusters)
  hmm_old = hmm;
  hmm = vbhmm_standardize(hmm, vbopt.sortclusters);
  %[wsort, wi] = sort(hmm.N, 1, 'descend');
  %hmm = vbhmm_permute(hmm, wi);
end

if (vbopt.showplot)
  vbhmm_plot(hmm, data);
end

% append the options
hmm.vbopt = vbopt;
  
function vbopt = setdefault(vbopt, field, value)
if ~isfield(vbopt, field)
  vbopt.(field) = value;
end
