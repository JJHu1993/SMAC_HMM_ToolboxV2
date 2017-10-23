function [fbstats] = vbhmm_fb(data, hmm_varpar, opt)
% vbhmm_fb - run forward-backward algorithm
%
% this is an internal function called by vbhmm_em
%
% [fbstats] = vbhmm_fb(data, hmm_varpar, opt)
%
% opt.savexi = 1: also save xi for each t 
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


% hmm parameters: v, W, epsilon, alpha, m, beta
v       = hmm_varpar.v;
W       = hmm_varpar.W;
epsilon = hmm_varpar.epsilon;
alpha   = hmm_varpar.alpha;
m       = hmm_varpar.m;
beta    = hmm_varpar.beta;

% options: group_map, usegroups
usegroups = opt.usegroups;
if usegroups
  group_map = opt.group_map;
  numgroups = opt.numgroups;
end

% other options: save xi
if isfield(opt, 'savexi')
  savexi = opt.savexi;
else
  savexi = 0;
end

% some constants: K, dim, N, maxT, lengthT, const, const_denominator
K   = size(m, 2);
dim = size(m, 1);
N   = length(data);
maxT    = max(cellfun('prodofsize', data)/dim);
lengthT = maxT;
const = dim*log(2);
const_denominator = (dim*log(2*pi))/2;

% pre-calculate constants
for k = 1:K
  t1 = psi(0, 0.5*repmat(v(k)+1,dim,1) - 0.5*[1:dim]');
  logLambdaTilde(k) = sum(t1) + const  + log(det(W(:,:,k)));
end

if ~usegroups
  for k = 1:K
    %Bishop (10.66)
    psiEpsilonHat(k) = psi(0,sum(epsilon(:,k)));
    logATilde(:,k) = psi(0,epsilon(:,k)) - psiEpsilonHat(k);  % A(i,j) = p(j->i) [column]
  end
  psiAlphaHat = psi(0,sum(alpha));
  logPiTilde = psi(0,alpha) - psiAlphaHat;
  
else
  psiEpsilonHat = {};   logATilde = {};
  psiAlphaHat = {};     logPiTilde = {};
  for g = 1:numgroups
    for k = 1:K
      %Bishop (10.66)
      psiEpsilonHat{g}(k) = psi(0,sum(epsilon{g}(:,k)));
      logATilde{g}(:,k) = psi(0,epsilon{g}(:,k)) - psiEpsilonHat{g}(k);  % A(i,j) = p(j->i) [column]
    end
    psiAlphaHat{g} = psi(0,sum(alpha{g}));
    logPiTilde{g} = psi(0,alpha{g}) - psiAlphaHat{g};
  end
end

logrho_Saved = zeros(K, N, lengthT);
fb_qnorm = zeros(1,N);

gamma_sum = zeros(K,N,maxT);
sumxi_sum = zeros(K,K,N);

if savexi
  xi_Saved = cell(1,N);
  for n=1:N
    xi_Saved{n} = zeros(K,K,size(data{n},1));
  end
end


for n = 1:N
  tdata = data{n}; tdata = tdata';
  tT = size(tdata,2);
  delta = []; logrho = [];
  
  delta = zeros(K,tT);
  
  for k = 1:K
    %OLD slow code Bishop (10.64)
    %for t = 1:tT
    %  diff = tdata(:,t) - m(:,k);
    %  delta(k,t) = dim/beta(k) + v(k)*diff'*W(:,:,k)*diff;
    %end
    %delta_old = delta;
    
    % ABC: 2016-04-21 - fast code
    diff = bsxfun(@minus, tdata, m(:,k));
    delta(k,:) = dim/beta(k) + v(k) * sum((W(:,:,k)*diff).*diff,1);
    
  end
  % OLD slow code - Bishop (10.46)
  %for k = 1:K
  %  for t = 1:tT
  %    logrho(k,t) = 0.5*logLambdaTilde(k) - 0.5*delta(k,t) - const_denominator;
  %  end
  %end
  %logrho_old = logrho;
  
  % ABC: 2016-04-21 - fast code
  logrho = bsxfun(@minus, 0.5*logLambdaTilde(:), 0.5*delta) - const_denominator;
  
  logrho_Saved(:,n,1:tT) = logrho;
  
  % forward_backward
  gamma = zeros(K,tT);
  sumxi = zeros(K,K);  % [row]
  
  fb_logrho = logrho';
  if ~usegroups
    fb_logPiTilde = logPiTilde';
    fb_logATilde = logATilde';
  else
    fb_logPiTilde = logPiTilde{group_map(n)}';
    fb_logATilde = logATilde{group_map(n)}';
  end
  t_alpha = [];t_beta = []; t_c = [];
  
  t_logPiTilde = exp(fb_logPiTilde); %priors
  t_logATilde = exp(fb_logATilde); %transitions
  t_logrho = exp(fb_logrho); %emissions
  t_x = tdata';
  t_T = size(t_x,1);
  if t_T >= 1
    %forward
    t_gamma = [];
    t_sumxi = zeros(K,K);
    t_alpha(1,:) = t_logPiTilde.*t_logrho(1,:);
    
    % 2016-04-29 ABC: rescale for numerical stability (otherwise values get too small)
    t_c(1) = sum(t_alpha(1,:));
    t_alpha(1,:) = t_alpha(1,:) / t_c(1);
    
    if t_T > 1
      for i=2:t_T
        t_alpha(i,:) = t_alpha(i-1,:)*t_logATilde.*t_logrho(i,:);
        
        % 2016-04-29 ABC: rescale for numerical stability
        t_c(i) = sum(t_alpha(i,:));
        t_alpha(i,:) = t_alpha(i,:) / t_c(i);
      end;
    end
    
    %backward
    t_beta(t_T,:) = ones(1,K)./K;
    t_gamma(:,t_T) = (t_alpha(t_T,:).*t_beta(t_T,:))';
    
    if t_T > 1
      for i=(t_T-1):-1:1
        bpi = (t_beta(i+1,:).*t_logrho(i+1,:));
        t_beta(i,:) = bpi*t_logATilde';
        
        % 2016-04-29 ABC: rescale
        t_beta(i,:) = t_beta(i,:)/t_c(i+1);
        
        t_gamma(:,i) = (t_alpha(i,:).*t_beta(i,:))';
        
        %t_sumxi = t_sumxi + (t_logATilde.*(t_alpha(i,:)'*bpi));
        
        % 2016-04-29 ABC: rescale xi
        tmp_xi = (t_logATilde.*(t_alpha(i,:)'*bpi)) / t_c(i+1);
        
        % 2016-04-29 ABC BUG FIX: normalize xi matrix to sum to 1
        % (it's a joint probability matrix)
        tmp_xi = tmp_xi / sum(tmp_xi(:));
        
        % accumulate
        t_sumxi = t_sumxi + tmp_xi;
        
        if savexi
          xi_Saved{n}(:,:,i) = tmp_xi;
        end
      end
    end
    for i = 1:size(t_gamma,2)
      gamma(:,i) = t_gamma(:,i);
    end
    sumxi = t_sumxi;
  end
  gamma_sum(:,n,1:tT) = gamma;
  sumxi_sum(:,:,n) = sumxi;
  
  % from scaling constants
  fb_qnorm(n) = sum(log(t_c));  
end

% output
fbstats.logrho_Saved = logrho_Saved;
fbstats.gamma_sum = gamma_sum;
fbstats.sumxi_sum = sumxi_sum;
fbstats.fb_qnorm  = fb_qnorm;
fbstats.logLambdaTilde = logLambdaTilde;
fbstats.logPiTilde = logPiTilde;
fbstats.logATilde = logATilde;

if savexi
  fbstats.xi_Saved = xi_Saved;
end

