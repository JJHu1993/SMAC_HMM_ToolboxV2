function [mix_t] = vbhmm_init(datai,K,ini)
% vbhmm_init - initialize vbhmm (internal function)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

VERBOSE_MODE = ini.verbose;

%data = convert_to_matrix(datai);
data = cat(1,datai{:});
[N,dim] = size(data);

switch(ini.initmode)
  
  %% initialize GMM with random initialization
  case 'random'
    
    try
      warning('off', 'stats:gmdistribution:FailedToConverge');
      
      if ~isempty(ini.random_gmm_opt)
        if (VERBOSE_MODE >= 3)
          fprintf('random gmm init: passing random_gmm_opt = ');
          ini.random_gmm_opt
        end
      end
      
      % use the Matlab fit function to find the GMM components
      % 2017-01-21 - ABC - added ability to pass other options (for Antoine)
      try
          mix = gmdistribution.fit(data,K, ini.random_gmm_opt{:}, 'Options', struct('TolFun', 1e-5));
      catch
          mix = gmdistribution.fit(data,K,'CovType','diagonal','SharedCov',true,'Regularize',0.0001, 'Options', struct('TolFun', 1e-5));
      end
      
    catch ME
      
      if strcmp(ME.identifier, 'stats:gmdistribution:IllCondCovIter')
        if (VERBOSE_MODE >= 2)
          fprintf('using shared covariance');
        end
        mix = gmdistribution.fit(data,K, 'SharedCov', true, 'Options', struct('TolFun', 1e-5));
        % SharedCov -> SharedCovariance
        
      else
        
        % otherwise use our built-in function (if Stats toolbox is not available)
        if (VERBOSE_MODE >= 2)
          fprintf('using built-in GMM');
        end
        gmmopt.cvmode = 'full';
        gmmopt.initmode = 'random';
        gmmopt.verbose = 0;
        gmmmix = gmm_learn(data', K, gmmopt);
        mix.PComponents = gmmmix.pi(:)';
        mix.mu = cat(2, gmmmix.mu{:})';
        mix.Sigma = cat(3, gmmmix.cv{:});
      end
    end
    
    %% initialize GMM with given GMM
  case 'initgmm'
    mix.Sigma = cat(3, ini.initgmm.cov{:});    
    mix.mu = cat(1, ini.initgmm.mean{:});    
    if size(mix.mu, 1) ~= K
      error('bad initgmm dimensions -- possibly mean is not a row vector');
    end
    
    mix.PComponents = ini.initgmm.prior(:)';
  
    %% initialize GMM with component splitting
  case 'split'
    gmmopt.cvmode = 'full';
    gmmopt.initmode = 'split';
    gmmopt.verbose = 0;
    gmmmix = gmm_learn(data', K, gmmopt);
        
    mix.PComponents = gmmmix.pi(:)';
    mix.mu = cat(2, gmmmix.mu{:})';
    mix.Sigma = cat(3, gmmmix.cv{:});
    
  otherwise
    error('bad initmode');
end

mix_t = {};
mix_t.dim = dim;
mix_t.K = K;
mix_t.N = N;

% setup hyperparameters
mix_t.alpha0   = ini.alpha;
mix_t.epsilon0 = ini.epsilon;
if length(ini.mu) ~= dim
  error(sprintf('vbopt.mu should have dimension D=%d', dim));
end
mix_t.m0       = ini.mu;
mix_t.beta0    = ini.beta;
if numel(ini.W) == 1
  % isotropic W
  mix_t.W0       = ini.W*eye(dim);  % 2016/04/26: ABC BUG Fix: was inv(ini.W*eye(dim))
else
  % diagonal W
  if numel(ini.W) ~= dim
    error(sprintf('vbopt.W should have dimension D=%d for diagonal matrix', dim));
  end
  mix_t.W0       = diag(ini.W);
end
if ini.v<=dim-1
  error('v not large enough');
end
mix_t.v0       = ini.v; % should be larger than p-1 degrees of freedom (or dimensions)
mix_t.W0inv    = inv(mix_t.W0);

% setup model (M-step)
mix_t.Nk = N*mix.PComponents';
mix_t.Nk2 = N*repmat(1/K,K,1);

mix_t.xbar = mix.mu';

if size(mix.Sigma,3) == K
  mix_t.S = mix.Sigma;
elseif (size(mix.Sigma,3) == 1)
  % handle shared covariance
  mix_t.S = repmat(mix.Sigma, [1 1 K]);
end

% 2017-01-21: handle diagonal Sigma (for Antoine)
% for diagonal covarainces, one of the dimensions will have size 1
if ((size(mix_t.S,1) == 1) || (size(mix_t.S,2)==1)) && (dim > 1)
  oldS = mix_t.S;
  mix_t.S = zeros(dim, dim, K);
  for j=1:K
    mix_t.S(:,:,j) = diag(oldS(:,:,j)); % make the full covariance
  end
end

mix_t.alpha = mix_t.alpha0 + mix_t.Nk2;
for k = 1:K
    mix_t.epsilon(:,k) = mix_t.epsilon0 + mix_t.Nk2;
end
mix_t.beta = mix_t.beta0 + mix_t.Nk;
mix_t.v = mix_t.v0 + mix_t.Nk + 1;   % 2016/04/26: BUG FIX (add 1)
mix_t.m = ((mix_t.beta0*mix_t.m0)*ones(1,K) + (ones(dim,1)*mix_t.Nk').*mix_t.xbar)./(ones(dim,1)*mix_t.beta');
mix_t.W = zeros(dim,dim,K);
for k = 1:K
    mult1 = mix_t.beta0.*mix_t.Nk(k)/(mix_t.beta0 + mix_t.Nk(k));
    diff3 = mix_t.xbar(:,k) - mix_t.m0;    
    mix_t.W(:,:,k) = inv(mix_t.W0inv + mix_t.Nk(k)*mix_t.S(:,:,k) + mult1*diff3*diff3');
end  
mix_t.C = zeros(dim,dim,K);
mix_t.const = dim*log(2);
mix_t.const_denominator = (dim*log(2*pi))/2;

%mix_t.maxIter = ini.maxIter;
%mix_t.minDiff = ini.minDiff;

%convert struct to matrix
%function out = convert_to_matrix(data)
%out = []; j = 1;
%for i = 1:size(data,1)
%    tout = data{i,1};
%    k = size(tout,1);
%    out(j:j+k-1,:) = tout;
%    j = j + k;
%end