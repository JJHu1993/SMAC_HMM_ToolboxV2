% vbgmm_learn - learn GMM with Variational Bayesian EM algorithm
%
%   [vb_gmm] = vbgmm_learn(data, k, vbopt)
%
% INPUTS
%   data = [NxD] matrix: N = samples, D = dimension
%      k = scalar: number of clusters
%          vector: automatically selects the number of clusters from the given values. 
%                  The model k with highest log-likelihood is selected.
%
%  vbopt = structure containing other options as below:
%
%   VB hyper-parameters:
%     vbopt.alpha = Dirichlet distribution concentration parameter -- large value
%                   encourages uniform prior, small value encourages concentrated prior (default=0.1)
%     vbopt.mu    = prior mean (default = [256;192])
%     vbopt.W     = size of the inverse Wishart distribution (default=0.005) -- 
%                   i.e., inverse covariance of prior ROI size is vbopt.W*I
%     vbopt.v     = dof of Wishart, v > D-1. (default=10)
%     vbopt.beta  = Wishart concentration parameter -- large value encourages use of
%                   prior mean & W, while small value ignores them. (default=1)
%  
%   EM Algorithm parameters
%     vbopt.initmode     = initialization method (default='random')
%                            'random' - initialize emissions using GMM with random initialization
%                            'initgmm' - specify a GMM for the emissions.
%     vbopt.numtrials    = number of trails for 'random' initialization (default=1)
%     vbopt.initgmm      = initial GMM for 'initgmm' 
%     vbopt.maxIter      = max number of iterations (default=30)
%     vbopt.minDiff      = tolerence for convergence (default=1e-5)
%     vbopt.showplot     = show plots (default=0)
%     vbopt.sortclusters = '' - no sorting [default]
%                          'd' - sort clusters by descending number of samples
%
% OUTPUT
%   vb_gmm.m       = means [D x K]
%   vb_gmm.C       = covariances [D x D x K]
%   vb_gmm.weight  = component weights [1 x K]
%   vb_gmm.LL      = log-likelihood of data
%   vb_gmm.iter    = number of iterations run
%   vb_gmm.vbopt   = vbopt options structure
%   vb_gmm.initmix = the initial GMM
%   vb_gmm.r       = responsibilities [N x K] -- probability of each data sample belonging to a cluster.
%  
%  for model selection:
%   vb_gmm.model_LL    = log-likelihoods for all models tested
%   vb_gmm.model_k     = the K-values used
%   vb_gmm.model_bestK = the K with the best value.
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


% TODO:
% x- expose more parameters
% x - specify init gmm
% x - random init w/ multiple trials
% - parameter description

% Versions
% 2016-04-18: - code from Tim (adapted from gmmVBEM3 by Emtiyaz, CS, UBC, June 2007)
% 2016-04-19: - added documentation for equations
%             - fixed bug with v: missing +1
%             - added numtrials for selecting best from random initialization
%             - added option for plotting
%             - added minDiff option
% 2016-04-20: - added model selection when specifying a set of k values
% 2016-04-25: - BUG fix in selection of best random initialization (random init works now)
%             - option to sort clusters
% 2016-04-26: - make W prior a scalar input (scalar*eye)

function vb_gmm = vbgmm_learn(data, k, vbopt)

[N,dim] = size(data);

if nargin<3
  vbopt = struct;
end

vbopt = setdefault(vbopt, 'alpha', 0.1);  
vbopt = setdefault(vbopt, 'mu', [256;192]);
vbopt = setdefault(vbopt, 'beta', 1);
vbopt = setdefault(vbopt, 'W', .005);
vbopt = setdefault(vbopt, 'v', 10);
vbopt = setdefault(vbopt, 'initmode', 'random');
vbopt = setdefault(vbopt, 'numtrials', 1);
vbopt = setdefault(vbopt, 'maxIter', 30);
vbopt = setdefault(vbopt, 'minDiff',  1e-5);
vbopt = setdefault(vbopt, 'showplot', 1);
vbopt = setdefault(vbopt, 'sortclusters', '');

% error checks
if (vbopt.v <= dim-1)
  error('Parameter v should be > D-1');
end


%% run for multiple K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if length(k)>1
  % turn off plotting
  vbopt2 = vbopt;
  vbopt2.showplot = 0;  
  
  % call learning for each value of k
  out_all = cell(1,length(k));
  LLk_all  = zeros(1,length(k));
  for ki = 1:length(k)
    fprintf('-- k=%d --\n', k(ki));
    
    % for initgmm, select one
    if strcmp(vbopt2.initmode, 'initgmm')
      vbopt2.initgmm = vbopt.initgmm{ki};
    end
    
    % call learning with a single k
    out_all{ki} = vbgmm_learn(data, k(ki), vbopt2);
    LLk_all(ki) = out_all{ki}.LL;
  end

  % correct for multiple parameterizations
  LLk_all = LLk_all + gammaln(k+1)
  
  % get K with max data likelihood
  [maxLLk,ind] = max(LLk_all);
  
  % return the best model
  vb_gmm       = out_all{ind};
  vb_gmm.model_LL    = LLk_all;
  vb_gmm.model_k     = k;
  vb_gmm.model_bestK = k(ind);
  
  fprintf('best model: k=%d; L=%g\n', k(ind), maxLLk);
  
  if vbopt.showplot
    figure
    hold on
    plot(k, LLk_all, 'bx-')
    plot([min(k), max(k)], [maxLLk, maxLLk], 'k--');
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
      numits = vbopt.numtrials;
      vb_gmms = cell(1,numits);
      mix     = cell(1,numits);
      LLall   = zeros(1,numits);
      % run several iterations
      for it = 1:numits
        mix{it} = gmdistribution.fit(data, k, 'Options', struct('TolFun', 1e-5));
        vb_gmms{it} = VBEM(data, mix{it}, vbopt);
        LLall(it) = vb_gmms{it}.LL;
      end

      % choose the best
      [maxLL,maxind] = max(LLall);
      LLall
      maxLL
      maxind
      vb_gmm = vb_gmms{maxind};   % ABC - bug fix 2016/04/25   
      
    %%% Initialize with learned GMM %%%%%%%%%%%%%%%%%%%%%%
    case 'initgmm'
      mix.NComponents = vbopt.initgmm.K;
      mix.PComponents = vbopt.initgmm.pi;
      mix.mu          = [vbopt.initgmm.mu{:}]';
      mix.Sigma       = cat(3,vbopt.initgmm.cv{:});
 
      vb_gmm = VBEM(data, mix, vbopt);
  end


end

% post processing
% - reorder based on cluster size
if (vbopt.sortclusters == 'd')
  vb_gmmold = vb_gmm;
  [wsort, wi] = sort(vb_gmm.weight(:), 1, 'descend');
  
  % reorder posterior matrix
  vb_gmm.r = vb_gmmold.r(:,wi);
  
  % reorder GMM components
  vb_gmm.weight = vb_gmmold.weight(wi);
  vb_gmm.m = vb_gmmold.m(:,wi);
  vb_gmm.C = vb_gmmold.C(:,:,wi);
end

% show the plot of the final result
if vbopt.showplot
  plot_vbgmm(vb_gmm,data)
end

return

%%% Function for running the actual VB algorithm %%%%%%%%%%%%%%%%%%%%%%
function out = VBEM(x,mix,vbopt)
K = mix.NComponents;
x = x'; 
[D N] = size(x);
L = -flintmax;
lastL = -flintmax;
minDiff = vbopt.minDiff;

% create variables
logLambdaTilde = zeros(1,K);
E = zeros(N,K);
trSW = zeros(1,K);
xbarWxbar = zeros(1,K);
mWm = zeros(1,K);
trW0invW = zeros(1,K);
C = zeros(D,D,K);

% hyperparameters
alpha0 = vbopt.alpha;
m0 = vbopt.mu;
beta0 = vbopt.beta;
W0 = vbopt.W * eye(D);
W0inv = inv(W0);
v0 = vbopt.v;
Nk = N*mix.PComponents';
xbar = mix.mu';
S = mix.Sigma;

% initialization using the provided GMM
alpha = alpha0 + Nk;   % (PRML 10.58)
beta = beta0 + Nk;     % (PRML 10.60)
v = v0 + Nk + 1;       % (PRML 10.63) (ABC: bug fix: was v=v0+Nk)
m = ((beta0*m0)*ones(1,K) + (ones(D,1)*Nk').*xbar)./(ones(D,1)*beta');  % mean (PRML 10.61)
W = zeros(D,D,K);
for k = 1:K
    % Wishart (PRML 10.62)
    mult1 = beta0.*Nk(k)/(beta0 + Nk(k));
    diff3 = xbar(:,k) - m0;
    W(:,:,k) = inv(W0inv + Nk(k)*S(:,:,k) + mult1*diff3*diff3');
end  

% VB-EM iterations
for iter = 1:vbopt.maxIter
  %% E-step
  psiAlphaHat = psi(0,sum(alpha));
  % PRML 10.66  
  logPiTilde = psi(0,alpha) - psiAlphaHat;
  const = D*log(2);
  for k = 1:K
    % PRML 10.65   
    t1 = psi(0, 0.5*repmat(v(k)+1,D,1) - 0.5*[1:D]');
    logLambdaTilde(k) = sum(t1) + const  + log(det(W(:,:,k)));
        
    % Expected distance (PRML, 10.64) - OLD code
    %for n = 1:N
    %  diff = x(:,n) - m(:,k);
    %  E(n,k) = D/beta(k) + v(k)*diff'*W(:,:,k)*diff;
    %end
    
    % Expected distance (PRML, 10.64) - FAST code
    abc_diff = bsxfun(@minus, x, m(:,k));
    abc_E = D/beta(k) + v(k)* sum((W(:,:,k)*abc_diff).*abc_diff,1);
    E(:,k) = abc_E';    
  end
  
  % calculate responsibilities (PRML, 10.46, 10.49, 10.67)
  logRho = repmat(logPiTilde' + 0.5*logLambdaTilde, N,1) - 0.5*E;
  logSumRho = log(sum(exp(logRho),2));
  logr = logRho - repmat(logSumRho, 1,K);
  r = exp(logr);
  
  % size of each cluster (PRML 10.51)
  Nk = exp(log(sum(exp(logr),1)))';
  Nk = Nk + 1e-10;
  
  for k=1:K
    % calculate statistics (PRML 10.52, 10.53)
    xbar(:,k) = sum(repmat(r(:,k)',D,1).*x,2)/Nk(k);
    diff1 = x - repmat(xbar(:,k),1,N);
    diff2 = repmat(r(:,k)',D,1).*diff1;
    S(:,:,k) = (diff2*diff1')./Nk(k);
  end
  
  %% variational lower bound (calculated before updates)
  
  % constants
  logCalpha0 = gammaln(K*alpha0) - K*gammaln(alpha0);
  logB0 = (v0/2)*log(det(W0inv)) - (v0*D/2)*log(2) ...
          - (D*(D-1)/4)*log(pi) - sum(gammaln(0.5*(v0+1 -[1:D])));
  logCalpha = gammaln(sum(alpha)) - sum(gammaln(alpha));
  H =0;
  for k = 1:K
    % 
    logBk = -(v(k)/2)*log(det(W(:,:,k))) - (v(k)*D/2)*log(2)...
            - (D*(D-1)/4)*log(pi) - sum(gammaln(0.5*(v(k) + 1 - [1:D])));
    H = H -logBk - 0.5*(v(k) -D-1)*logLambdaTilde(k) + 0.5*v(k)*D;
    trSW(k) = trace(v(k)*S(:,:,k)*W(:,:,k));
    diff = xbar(:,k) - m(:,k);
    xbarWxbar(k) = diff'*W(:,:,k)*diff;
    diff = m(:,k) - m0;
    mWm(k) = diff'*W(:,:,k)*diff; 
    trW0invW(k) = trace(W0inv*W(:,:,k));
  end
  % PRML 10.71
  Lt1 = 0.5*sum(Nk.*(logLambdaTilde' - D./beta...
        - trSW' - v.*xbarWxbar' - D*log(2*pi)));
  % PRML 10.72
  Lt2 = sum(Nk.*logPiTilde);
  % PRML 10.73
  Lt3 = logCalpha0 + (alpha0 -1)*sum(logPiTilde);
  % PRML 10.74
  Lt41 = 0.5*sum(D*log(beta0/(2*pi)) + logLambdaTilde' - D*beta0./beta - beta0.*v.*mWm');
  Lt42 = K*logB0 + 0.5*(v0-D-1)*sum(logLambdaTilde) - 0.5*sum(v.*trW0invW');
  Lt4 = Lt41+Lt42;
  % PRML 10.75
  Lt5 = sum(sum(r.*logr));
  % PRML 10.76
  Lt6 = sum((alpha - 1).*logPiTilde) + logCalpha;
  % PRML 10.77
  Lt7 = 0.5*sum(logLambdaTilde' + D.*log(beta/(2*pi))) - 0.5*D*K - H;

  %% M-step parameter updates
  alpha = alpha0 + Nk;   % (PRML 10.58)
  beta = beta0 + Nk;     % (PRML 10.60)
  v = v0 + Nk + 1;       % (PRML 10.63) (ABC: bug fix: was v=v0+Nk)
  m = (repmat(beta0.*m0,1,K) + repmat(Nk',D,1).*xbar)./repmat(beta',D,1);  % means (PRML 10.61)
  for k = 1:K
    % Wishart matrix (PRML 10.62)
    mult1 = beta0.*Nk(k)/(beta0 + Nk(k));
    diff3 = xbar(:,k) - m0;
    W(:,:,k) = inv(W0inv + Nk(k)*S(:,:,k) + mult1*diff3*diff3');
  end
  
  % calculate covariance matrices 
  for i = 1:K
    % The mean of the inverse-wishart distribution 
    % with parameter inv(W) and dof v(i)
    C(:,:,i) = inv(W(:,:,i))/(v(i)-D-1);
  end
  
  %% Calculate variational lower-bound, stop if converged
  lastL = L;
  % (PRML 10.70)
  L = Lt1 + Lt2 + Lt3 + Lt4 - Lt5 - Lt6 - Lt7;
  if iter > 1
    likIncr = abs((L-lastL)/lastL);
    fprintf('%d: L=%g; dL=%g', iter, L, likIncr);
    if (L-lastL < 0)
      fprintf(' !!!');
    end
    fprintf('\n');
    if likIncr <= minDiff
      break;
    end
  end
  
  if (iter == vbopt.maxIter)
    warning(sprintf('did not converge in %d iterations', iter));
  end
end

fprintf('%d: L=%g; dL=%g\n', iter, L, likIncr);

out.m = m;
out.C = C;
out.weight = alpha/sum(alpha);
out.LL = L;
out.iter = iter;
out.vbopt = vbopt;
out.initmix  = mix;
out.r = r;


%% plotting functions
function plot_vbgmm(vb_gmm,data)
color = ['r','g','b','y', 'm', 'k', 'c'];
figure
subplot(1,2,1)
axis ij
hold on
[~,rs] = max(vb_gmm.r,[],2);
for k=1:size(vb_gmm.m,2)
  ii = find(rs==k);
  scatter(data(ii,1),data(ii,2), [color(mod(k-1,length(color))+1) '.']);
end
for k = 1:size(vb_gmm.m,2)
    mu = vb_gmm.m(:,k);
    sigma(:,:) = vb_gmm.C(1:2,1:2,k);
    weight = vb_gmm.weight(k);
    plot2D(weight,mu,sigma,color(1,mod(k-1,length(color))+1),k)
end
for k=1:size(vb_gmm.m,2)
   mu = vb_gmm.m(:,k);
   text(mu(1), mu(2), sprintf('%d', k), 'Color', 'k');
end
hold off
subplot(1,2,2)
bar(1:size(vb_gmm.m,2), vb_gmm.weight);
xlabel('weights');

function plot2D(weight,mu,Sigma,color,kcomp)

mu = mu(:);
[U,D] = eig(Sigma);
n = 100;
t = linspace(0,2*pi,n);
xy = [cos(t);sin(t)];
k = sqrt(conf2mahal(0.95,2));
w = (k*U*sqrt(D))*xy;
z = repmat(mu,[1 n])+w;
if (weight < 0.001)
  lstyle = ':';
else
  lstyle = '-';
end
h = plot(z(1,:),z(2,:),'LineStyle', lstyle, 'Color',color, 'LineWidth', 1);
%h = text(mu(1), mu(2), sprintf('%d', kcomp), 'Color', 'k');

function m = conf2mahal(c,d)
m = chi2inv(c,d);

function vbopt = setdefault(vbopt, field, value)
if ~isfield(vbopt, field)
  vbopt.(field) = value;
end

