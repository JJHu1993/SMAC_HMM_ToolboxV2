function [LL, LLcomp, post] = gmm_ll(X, gmm)
% gmm_ll - log-likelihood of GMM
%
% USAGE
%   [LL, LLcomp, post] = gmm_ll(X, gmm)
%
% INPUTS
%   X = [x1 ... xn]  (column vectors)
%   gmm  = GMM model (see gmm_learn)
%
% OUTPUTS
%   LL     = log-likelihoods of X      [n x 1]
%   LLcomp = component log-likelihoods [n x K]
%   post   = posterior probabilities   [n x K]
%
% if bkgndclass is used, LLcomp and post are [n x (K+1)], where
% the last column is the log-likelihood and posterior in the background class
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 3/5/2012 - ABC - speed update using bsxfun instead of repmat
% 3/7/2012 - ABC, Alvin (Asa) - add background class
% 11/05/2012 - ABC - break up calculation if large K and only calculating LL

K = gmm.K;
d = size(X,1);
N = size(X,2);

if (K > 200) && (nargout == 1)
  % if only looking for LL, and K large
  %% calculate LL only %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % find the max
  LLmax = zeros(N,1);
  for c=1:K
    tmpLL = logComponent(gmm, X, c, d, N);
    LLmax = max(LLmax, tmpLL);
  end
  
  if isfield(gmm, 'bkgndclass') && (gmm.bkgndclass ~= 0)
    error('not supported');
  end
  
  % do logtrick
  cterm = zeros(N,1);
  for c=1:K
    tmpLL = logComponent(gmm, X, c, d, N);
    cterm = cterm + exp(tmpLL - LLmax);
  end
  
  LL = LLmax + log(cterm);

  % sanity check
  if 0
    LLcomp = zeros(N,K);
    % for each class
    for c=1:K
      LLcomp(:,c) = logComponent(gmm, X, c, d, N);
    end
    LL2 = logtrick2(LLcomp);
    totalerror(LL, LL2)
  end
  
else
  %% calculate LL and LLcomp %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % component log-likelihood
  LLcomp = zeros(N,K);

  % for each class
  for c=1:K
    LLcomp(:,c) = logComponent(gmm, X, c, d, N);
  end
  
  % add background class
  if isfield(gmm, 'bkgndclass') && (gmm.bkgndclass ~= 0)
    LLcomp(:,end+1) = log(gmm.bkgndclass);
  end

  LL = logtrick2(LLcomp);
end


%% output posterior %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (nargout>2)
  %post = exp(LLcomp - repmat(LL,1,K));
  post = exp(bsxfun(@minus, LLcomp, LL));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% return the log-likelihood of component c
function myLLcomp = logComponent(gmm, X, c, d, N)
% initialize
myLLcomp = zeros(N,1);

% setup pdf constants
mu = gmm.mu{c};
cv = gmm.cv{c};
  
%tempx = (X - repmat(mu, 1, N));
tempx = bsxfun(@minus, X, mu);
  
switch(gmm.cvmode)
  case 'iid'
    g  = sum(tempx.*tempx,1) / cv;
    ld = d*log(cv);
    
  case 'diag'
    %g  = sum((repmat(1./cv(:),1,N).*tempx).*tempx,1);
    
    % this is not as numerically stable, but much faster
    %g = sum(bsxfun(@times, tempx.*tempx, 1./cv(:)), 1);
    
    % this is more numerically stable.
    tmp = bsxfun(@times, tempx, 1./sqrt(cv(:)));
    g = sum(tmp.*tmp, 1);
    
    ld = sum(log(cv));
    
   case 'full'
    g  = sum((inv(cv)*tempx).*tempx,1);
    %ld = log(det(cv));
    ld = logdet(cv);
    
  otherwise
    error('bad mode');
end

myLLcomp = -0.5*g' - (d/2)*log(2*pi) - 0.5*ld + log(gmm.pi(c));



function [s] = logtrick2(lA)
% logtrick - "log sum trick" - calculate log(sum(A)) using only log(A) 
%
%   s = logtrick(lA)
%
%   lA = log sum is calculated over each row
% 
[mv, mi] = max(lA, [], 2);
%temp = lA - repmat(mv, size(lA,1), 1);
cterm = sum(exp(bsxfun(@minus, lA, mv)), 2);
s = mv + log(cterm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [s] = logtrick(lA)
% logtrick - "log sum trick" - calculate log(sum(A)) using only log(A) 
%
%   s = logtrick(lA)
%
%   lA = column vector of log values
%
%   if lA is a matrix, then the log sum is calculated over each column
% 
%  6-20-2004 - don't sort lA, just find the largest value.

%[A2] = sort(lA);
%cterm = sum( exp( A2(1:end-1) - A2(end)) );
%s = A2(end) + log(1 + cterm);

[mv, mi] = max(lA, [], 1);
temp = lA - repmat(mv, size(lA,1), 1);
cterm = sum(exp(temp),1);
s = mv + log(cterm);

%a2 = sort(lA, 1);
%temp = a2(1:end-1,:) - repmat(a2(end,:),size(a2,1)-1,1);
%cterm = sum(exp(temp),1);
%s = a2(end,:) + log(1+cterm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ld = logdet(A)

% R'*R = A, R is triangular
% det(R'*R) = det(A) = det(R)^2

%R = chol(A);
%ld = 2*sum(log(diag(R)));

[R, p] = chol(A);
if (p == 0)
  ld = 2*sum(log(diag(R)));
else
  x = eig(A);
  ld = sum(log(x));
  warning('logdet:chol', 'A is not PD for chol, using eig');
end

