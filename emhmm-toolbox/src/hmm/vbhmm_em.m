function [hmm,L] = vbhmm_em(data,K,ini)
% vbhmm_em - run EM for vbhmm (internal function)
% Use vbhmm_learn instead. For options, see vbhmm_learn.
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

VERBOSE_MODE = ini.verbose;

% get length of each chain
trial = length(data);
datalen = [];
for i = 1:length(data)
    d = data{i};
    datalen(i,1) = size(d,1);
end
lengthT = max(datalen); % find the longest chain
totalT  = sum(datalen);

%initialize the parameters
mix_t = vbhmm_init(data,K,ini); %initialize the parameters
mix = mix_t;
dim = mix.dim; %dimension of the data
K = mix.K; %no. of hidden states
N = trial; %no. of chains
maxT = lengthT; %the longest chain
alpha0 = mix.alpha0; %hyper-parameter for the priors
epsilon0 = mix.epsilon0; %hyper-parameter for the transitions
m0 = mix.m0; %hyper-parameter for the mean
beta0 = mix.beta0; %hyper-parameter for beta (Gamma)
v0 = mix.v0; %hyper-parameter for v (Inverse-Wishart)
W0inv = mix.W0inv; %hyper-parameter for Inverse-Wishart
alpha = mix.alpha; %priors
epsilon = mix.epsilon; %transitions
beta = mix.beta; %beta (Gamma)
v = mix.v; %v (Inverse-Wishart)
m = mix.m; %mean
W = mix.W; %Inverse-Wishart
C = mix.C; %covariance
const = mix.const; %constants
const_denominator = mix.const_denominator; %constants
maxIter = ini.maxIter; %maximum iterations allowed
minDiff = ini.minDiff; %termination criterion

L = -realmax; %log-likelihood
lastL = -realmax; %log-likelihood

% setup groups
if ~isempty(ini.groups)
  usegroups = 1;
  group_ids = unique(ini.groups);  % unique group ids
  numgroups = length(group_ids);
  for g=1:numgroups
    group_inds{g} = find(ini.groups == group_ids(g)); % indices for group members
  end
  % sanitized group membership (1 to G)
  group_map = zeros(1,length(ini.groups));
  for g=1:numgroups
    group_map(group_inds{g}) = g;
  end
  
  % reshape alpha, epsilon into cell
  % also Nk1 and M are cells
  tmp = epsilon;
  tmpa = alpha;
  epsilon = {};
  alpha = {};
  for g = 1:numgroups
    epsilon{g} = tmp;
    alpha{g} = tmpa;
  end
  
else
  usegroups = 0;
end

for iter = 1:maxIter
          
    %% E step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if 1      
      %% 2016-08-14 - call function for FB algorithm
      
      % setup HMM
      fbhmm_varpar.v = v;
      fbhmm_varpar.W = W;
      fbhmm_varpar.epsilon = epsilon;
      fbhmm_varpar.alpha = alpha;
      fbhmm_varpar.m = m;
      fbhmm_varpar.beta = beta;
      fbopt.usegroups = usegroups;
      if usegroups
        fbopt.group_map = group_map;
        fbopt.numgroups = numgroups;
      end
      
      % call FB algorithm
      [fbstats] = vbhmm_fb(data, fbhmm_varpar, fbopt);

      % get statistics and constants
      logrho_Saved = fbstats.logrho_Saved;
      gamma_sum = fbstats.gamma_sum;
      sumxi_sum = fbstats.sumxi_sum;
      fb_qnorm  = fbstats.fb_qnorm;
      logLambdaTilde = fbstats.logLambdaTilde;
      logPiTilde = fbstats.logPiTilde;
      logATilde = fbstats.logATilde;


    end
    
    if 0    
      %% OLD code - use inline code for FB algorithm
      
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
    end % end old code
    
    
    %% sum up the responsibilities
    
    % OLD slow code
    %t_Nk1 = zeros(K,maxT);
    %t_gamma_Saved = zeros(K,N,maxT);
    %for t = 1:maxT
    %  for n = 1:N        
    %    t_gamma = squeeze(gamma_sum(:,n,t));
    %    scale = sum(t_gamma); if scale == 0 scale = 1; end
    %    t_gamma = t_gamma./repmat(scale,K,1);
    %    t_gamma_Saved(:,n,t) = t_gamma;
    %    t_Nk1(:,t) = t_Nk1(:,t) + t_gamma;
    %  end
    %end
    
    % ABC: 2016-04-21 - fast code
    scale = sum(gamma_sum, 1);
    scale(scale==0) = 1;
    t_gamma_Saved = bsxfun(@rdivide, gamma_sum, scale);  % marginal responsibilities
    t_Nk1 = reshape(sum(t_gamma_Saved, 2), [K, maxT]);
    
    % for updating priors
    if ~usegroups
      Nk1 = t_Nk1(:,1);
      Nk1 = Nk1 + 1e-50;
    else
      for g=1:numgroups
        Nk1{g} = sum(t_gamma_Saved(:,group_inds{g},1), 2);
      end
    end
    
    % for updating beta and v
    Nk = sum(t_Nk1,2); 
    Nk = Nk + 1e-50;
    
    % OLD SLOW CODE
%     if 0
%       M = zeros(K,K);  % M(i,j) = soft counts i->j [row]
%       %t_sumxi_Saved = zeros(N,K,K);  % probability i->j [row]
% 
%       for n = 1:N     
%         %t_sumxi = squeeze(sumxi_sum(:,:,n));
%         %for k = 1:K
%         %  scale = sum(t_sumxi(k,:));
%         %  if scale == 0 scale = 1; end
%         %  t_sumxi(k,:) = t_sumxi(k,:)./repmat(scale,1,K);
%         %end
%         %t_sumxi_old = t_sumxi;
% 
%         % 2016-04-25: ABC - new fast code
%         t_sumxi = squeeze(sumxi_sum(:,:,n));
% 
%         % 2016-04-29: BUG FIX: do not need to normalize each row
%         %scale = sum(t_sumxi, 2);
%         %scale(scale==0) = 1;
%         %t_sumxi = bsxfun(@rdivide, t_sumxi, scale);
% 
%         %t_sumxi_Saved(n,:,:) = t_sumxi;
% 
%         M(:,:) = M(:,:) + t_sumxi; %for updating the transitions
%       end
%     end
    
    % 2016-04-29: ABC NEW fast code
    % M(i,j) = soft counts i->j [row]
    if ~usegroups
      M = sum(sumxi_sum, 3);      
    else
      for g=1:numgroups
        M{g} = sum(sumxi_sum(:,:,group_inds{g}),3);
      end
    end
    
    % OLD SLOW code
    %t_xbar = zeros(K,N,maxT,dim);
    %for k=1:K
    %  for n = 1:N
    %    x = data{n,1}; x = x'; tT = size(x,2);
    %    for t = 1:tT
    %      t_xbar(k,n,t,:) = t_gamma_Saved(k,n,t).*squeeze(x(:,t));
    %    end
    %  end
    %end
    %t_xbar_old = t_xbar;
    
    % ABC 2016-04-21 - fast code
    t_xbar = zeros(K,N,maxT,dim);
    for n=1:N
      x = data{n}; tT = size(x,1);
      t_xbar(:,n,1:tT,:) = bsxfun(@times, ...
        reshape(t_gamma_Saved(:,n,1:tT), [K,1,tT,1]), ...
        reshape(x, [1 1 tT dim]));
    end

    
    % OLD slow code
    %t1_xbar = zeros(K,N,dim);
    %for k = 1:K
    %  for n = 1:N
    %    for t = 1:maxT
    %      t1_xbar(k,n,:) = squeeze(t1_xbar(k,n,:)) + squeeze(t_xbar(k,n,t,:));
    %    end
    %  end
    %end

    % ABC 2016-04-21 - fast code
    t1_xbar = sum(t_xbar,3);
    
    % OLD slow code
    %t2_xbar = zeros(K,dim);
    %for k = 1:K
    %  for n = 1:N
    %    t2_xbar(k,:) = (squeeze(t2_xbar(k,:))' + squeeze(t1_xbar(k,n,:)))';
    %  end
    %end
    %t2_xbar_old = t2_xbar;
    
    % ABC 2016-04-21 - fast code
    t2_xbar = sum(t1_xbar,2);
    t2_xbar = reshape(t2_xbar, [K, dim]);

    % OLD slow code 
    %xbar = zeros(K,dim);
    %for k = 1:K
    %  for d = 1:dim
    %    xbar(k,d) = t2_xbar(k,d)/Nk(k); %for updating the means
    %  end
    %end
    %xbar_old = xbar;
    
    % ABC 2016-04-21 - fast code
    xbar = bsxfun(@rdivide, t2_xbar, Nk);
    
    % OLD slow code
    %diff1 = []; diff2 = []; t1_S = [];
    %for k = 1:K
    %  t_m = 1;
    %  for n = 1:N
    %    x = data{n,1}; 
    %    x = x'; 
    %    tT = size(x,2);
    %    for t = 1:tT
    %      diff1(k,:,t_m) = squeeze(x(:,t)) - xbar(k,:)';
    %      t_diff = squeeze(x(:,t)) - xbar(k,:)';
    %      diff2(k,:,t_m) = repmat(t_gamma_Saved(k,n,t),dim,1).*t_diff;
    %      t_m = t_m + 1;
    %    end
    %  end
    %  t_d1 = squeeze(diff1(k,:,:));
    %  t_d2 = squeeze(diff2(k,:,:));
    %  t_S = t_d2*t_d1';
    %  t_S = t_S./Nk(k);
    %  t1_S(:,:,k) = t_S; %for updating the covariances
    %end
    %t1_S_old = t1_S;
    
    % ABC 2016-04-21: fast code
    t1_S = zeros(dim, dim, K);
    for n=1:N
      x = data{n}';
      tT = size(x,2);
      
      for k=1:K
        d1 = bsxfun(@minus, x, xbar(k,:)');
        d2 = bsxfun(@times, reshape(t_gamma_Saved(k,n,1:tT), [1,tT]), d1);
        t1_S(:,:,k) = t1_S(:,:,k) + d1*d2';
      end
    end
    t1_S = bsxfun(@rdivide, t1_S, reshape(Nk, [1 1 K]));
    
    
    %% calculate the lower bound %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ABC (11)
    
    % constants
    logCalpha0 = gammaln(K*alpha0) - K*gammaln(alpha0);
    for k = 1:K
      logCepsilon0(k) = gammaln(K*epsilon0) - K*gammaln(epsilon0);
    end
    logB0 = (v0/2)*log(det(W0inv)) - (v0*dim/2)*log(2) ...
      - (dim*(dim-1)/4)*log(pi) - sum(gammaln(0.5*(v0+1 -[1:dim])));
    
    if ~usegroups      
      logCalpha = gammaln(sum(alpha)) - sum(gammaln(alpha));
      for k = 1:K
        logCepsilon(k) = gammaln(sum(epsilon(:,k))) - sum(gammaln(epsilon(:,k)));
      end
    else
      for g=1:numgroups
        logCalpha{g} = gammaln(sum(alpha{g})) - sum(gammaln(alpha{g}));
        for k = 1:K
          logCepsilon{g}(k) = gammaln(sum(epsilon{g}(:,k))) - sum(gammaln(epsilon{g}(:,k)));
        end
      end
    end
    
    H =0;
    for k = 1:K
      logBk = -(v(k)/2)*log(det(W(:,:,k))) - (v(k)*dim/2)*log(2)...
        - (dim*(dim-1)/4)*log(pi) - sum(gammaln(0.5*(v(k) + 1 - [1:dim])));
      H = H - logBk - 0.5*(v(k) - dim - 1)*logLambdaTilde(k) + 0.5*v(k)*dim;
      trSW(k) = trace(v(k)*t1_S(:,:,k)*W(:,:,k));
      xbarT = xbar(k,:)';
      diff = xbarT - m(:,k);
      xbarWxbar(k) = diff'*W(:,:,k)*diff;
      diff = m(:,k) - m0;
      mWm(k) = diff'*W(:,:,k)*diff;
      trW0invW(k) = trace(W0inv*W(:,:,k));
    end
    
    % E(log p(X|Z,mu,Lambda)  Bishop (10.71) - ABC term 1
    Lt1 = 0.5*sum(Nk.*(logLambdaTilde' - dim./beta...
      - trSW' - v.*xbarWxbar' - dim*log(2*pi))); 
    
    % initial responsibilities (t=1)
    gamma1 = t_gamma_Saved(:,:,1);
    
    % transition responsibilities
    %gamma_t1 = zeros(K,K,N);
    %for i = 1:size(logPiTilde,1)
    %  for k = 1:size(gamma1,1)
    %    for n = 1:size(gamma1,2)
    %      gamma_t1(i,k,n) = gamma1(k,n).*logPiTilde(i);
    %    end
    %  end
    %end
    
    % E[log p(Z|pi)]   Bishop (10.72) - ABC term 2, part 1
    if ~usegroups
      PiTilde_t = repmat(logPiTilde,1,N);
      gamma_t1 = gamma1.*PiTilde_t;
      Lt2a = sum(sum(gamma_t1));
    else
      Lt2a = 0;
      for g=1:numgroups
        PiTilde_t = logPiTilde{g};
        gamma_t1 = bsxfun(@times, gamma1(:,group_inds{g}), PiTilde_t);
        Lt2a = Lt2a + sum(sum(gamma_t1));
      end
    end
      

    % E[log p(Z|A)]   ~Bishop 10.72 - ABC term 2, part 2 [CORRECT?]
    if ~usegroups
      %sumxi_t1 = zeros(N,K,K);
      ATilde_t = logATilde';
      %for i = 1:size(t_sumxi_Saved,1)
      %  sumxi_t1(i,:,:) = squeeze(t_sumxi_Saved(i,:,:)).*ATilde_t;
      %end
      %Lt2b = sum(sum(sum(sumxi_t1)))
      Lt2b = sum(M(:).*ATilde_t(:));
    else
      Lt2b = 0;
      for g=1:numgroups
        ATilde_t = logATilde{g}';
        Lt2b = Lt2b + sum(M{g}(:).*ATilde_t(:));
      end
    end    
        
    % E[log p(Z|pi, A)] 
    % ABC term 2
    Lt2 = Lt2a + Lt2b;
    
    % E[log p(pi)]   Bishop (10.73)   ABC term 3
    if ~usegroups
      Lt3 = logCalpha0 + (alpha0-1)*sum(logPiTilde);
    else
      Lt3 = 0;
      for g=1:numgroups
        Lt3 = Lt3 + logCalpha0 + (alpha0-1)*sum(logPiTilde{g});    
      end
    end
    
    % E[log p(A)] = sum E[log p(a_j)]   (equivalent to Bishop 10.73) ABC term 4
    if ~usegroups
      for k = 1:K
        Lt4a(k) = logCepsilon0(k) + (epsilon0 -1)*sum(logATilde(:,k));
      end
      Lt4 = sum(Lt4a);   
    else
      Lt4 = 0;
      for g=1:numgroups
        for k = 1:K
          Lt4a(k) = logCepsilon0(k) + (epsilon0 -1)*sum(logATilde{g}(:,k));
        end
        Lt4 = Lt4 + sum(Lt4a);   
      end
    end
    
    % E[log p(mu, Lambda)]  Bishop (10.74)  ABC term 5
    Lt51 = 0.5*sum(dim*log(beta0/(2*pi)) + logLambdaTilde' - dim*beta0./beta - beta0.*v.*mWm');
    Lt52 = K*logB0 + 0.5*(v0-dim-1)*sum(logLambdaTilde) - 0.5*sum(v.*trW0invW');
    Lt5 = Lt51+Lt52; 
    
    % OLD CODE (Incorrect)
   % Lt51 = sum(Nk1.*log(Nk1));
   % logM = log(M);
   % Lt52 = M.*logM;
   % for i = 1:size(Lt52,1)
   %   for j = 1:size(Lt52,2)
   %     if isnan(Lt52(i,j)) == 1
   %       Lt52(i,j) = 0;
   %     end
   %   end
   % end    
   % Lt52 = sum(sum(Lt52));
   % Lt5 = Lt51 + Lt52; %Bishop (10.75)
    
    % 2016-04-26 ABC: use correct q(Z)
    
    % 2016-04-29 ABC: E[z log pi] (same as Lt2a)
    %Lt61 = sum(sum(bsxfun(@times, t_gamma_Saved(:,:,1), logPiTilde)));
    Lt61 = Lt2a;
    
    % 2016-04-29 ABC:  E[zt zt-1 log a]  (same as Lt2b)
    Lt62 = Lt2b;
    
    % 2016-04-29 ABC:  E[z log rho]
    % the zeros in logrho should remove times (tT+1):N
    Lt63 = sum(sum(sum(t_gamma_Saved.*logrho_Saved)));
    
    % 2016-04-29 ABC: normalization constant for q(Z)
    Lt64 = sum(fb_qnorm);
    %fprintf('   norm constant: %g\n', Lt64);
    
    % 2016-04-29 ABC: E[log q(Z)] - ABC term 6
    Lt6 = Lt61 + Lt62 + Lt63 - Lt64;
    
    % E[log q(pi)]  Bishop (10.76)  
    if ~usegroups
      Lt71 = sum((alpha - 1).*logPiTilde) + logCalpha;
    else
      Lt71 = 0;
      for g=1:numgroups
        Lt71 = Lt71 + sum((alpha{g} - 1).*logPiTilde{g}) + logCalpha{g};
      end
    end
    
    % E[log q(aj)]  (equivalent to Bishop 10.76)
    if ~usegroups
      for k = 1:K
        Lt72(k) = sum((epsilon(:,k) - 1).*logATilde(:,k)) + logCepsilon(k);
      end
      Lt72sum = sum(Lt72);
    else
      Lt72sum = 0;
      for g=1:numgroups
        for k = 1:K
          Lt72(k) = sum((epsilon{g}(:,k) - 1).*logATilde{g}(:,k)) + logCepsilon{g}(k);
        end
        Lt72sum = Lt72sum + sum(Lt72);
      end
    end
      
    % E[log q(pi, A)] - ABC term 7
    Lt7 = Lt71 + Lt72sum; 
    
    % E[q(mu,Lamba)]  Bishop (10.77) - ABC term 8  
    Lt8 = 0.5*sum(logLambdaTilde' + dim.*log(beta/(2*pi))) - 0.5*dim*K - H; 

    if iter > 1
      lastL = L;
    end
    L = Lt1 + Lt2 + Lt3 + Lt4 + Lt5 - Lt6 - Lt7 - Lt8;

    
    %% M step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % prior & transition parameters
    if ~usegroups      
      alpha = alpha0 + Nk1;
      epsilon = epsilon0 + M;
      epsilon = epsilon';      % [column]
    else
      for g=1:numgroups
        alpha{g} = alpha0 + Nk1{g};
        epsilon{g} = epsilon0 + M{g};
        epsilon{g} = epsilon{g}';      % [column]
      end
    end
    
    % update Gaussians
    if ~ini.fix_clusters
      % mean
      beta = beta0 + Nk;
      v = v0 + Nk + 1;
      for k = 1:K
        m(:,k) = (beta0.*m0 + Nk(k).*xbar(k,:)')./beta(k);
      end
      
      % wishart
      for k = 1:K
        if isempty(ini.fix_cov)
          mult1 = beta0.*Nk(k)/(beta0 + Nk(k));
          diff3 = xbar(k,:) - m0';
          diff3 = diff3';
          W(:,:,k) = inv(W0inv + Nk(k)*t1_S(:,:,k) + mult1*diff3*diff3');
          
        else
          % 2017-01-21 - fix the covariance matrix (for Antoine)
          % set Wishart W to the appropriate value to get the covariance matrix.
          W(:,:,k) = inv(ini.fix_cov)/(v(k)-dim-1);
        end
      end
    end
    
    % covariance
    for k = 1:K
      C(:,:,k) = inv(W(:,:,k))/(v(k)-dim-1);
    end
    
    if iter > 1
      likIncr = abs((L-lastL)/lastL);
      if (VERBOSE_MODE >= 3)
        fprintf('%d: L=%g; dL=%g', iter, L, likIncr);
        if (L-lastL < 0)
          fprintf(' !!!');
          keyboard
        end
        fprintf('\n');
      else 
        if (L-lastL < 0)
          if (VERBOSE_MODE >= 2)
            warning('LL decreased');
          end
        end
      end
      if likIncr <= minDiff
        break;
      end
    end


end

if (VERBOSE_MODE >= 2)
  fprintf('%d: L=%g; dL=%g\n', iter, L, likIncr);
end

if (VERBOSE_MODE >= 3)
  fprintf('-\n');
end

%generate the output model
% NOTE: if adding a new field, remember to modify vbhmm_permute.
hmm = {};
if ~usegroups
  prior_s = sum(alpha);
  prior = alpha ./ prior_s;
  hmm.prior = prior;
  trans_t = epsilon';  % [row]
  for k = 1:K
      scale = sum(trans_t(k,:));
      if scale == 0 scale = 1; end
      trans_t(k,:) = trans_t(k,:)./repmat(scale,1,K);
  end
  hmm.trans = trans_t;
else
  for g=1:numgroups
    prior_s = sum(alpha{g});
    prior = alpha{g} ./ prior_s;
    hmm.prior{g} = prior;
    
    trans_t = epsilon{g}';  % [row]
    for k = 1:K
      scale = sum(trans_t(k,:));
      if scale == 0 scale = 1; end
      trans_t(k,:) = trans_t(k,:)./repmat(scale,1,K);
    end
    hmm.trans{g} = trans_t;
  end
end

hmm.pdf = {};
for k = 1:K
  % Antoine mod - fix the covariance after learning
  %if ~isempty(ini.fix_cov)
  %  %C(:,:,k)=diag(diag(C(:,:,k)));%0s on 12 and 21
  %  C(:,:,k)=ini.fix_cov; %[ini.do_constrain_var_size 0; 0 ini.do_constrain_var_size];
  %end
  
  hmm.pdf{k,1}.mean = m(:,k)';
  hmm.pdf{k,1}.cov = C(:,:,k);
end
hmm.LL = L;
hmm.gamma = cell(1,N);
for n=1:N
  hmm.gamma{n} = reshape(t_gamma_Saved(:,n,1:datalen(n)), [K datalen(n)]);
end
hmm.M = M;     % transition counts
hmm.N1 = Nk1;  % prior counts
hmm.N  = Nk;   % cluster sizes

% save group info
if usegroups
  for g=1:numgroups
    ggamma = hmm.gamma(group_inds{g});
    hmm.Ng{g} = sum(cat(2, ggamma{:}), 2); % cluster size by group
  end
  hmm.group_map = group_map;
  hmm.group_ids = group_ids;
  hmm.group_inds = group_inds;
end

% save variational parameters
hmm.varpar.epsilon = epsilon;
hmm.varpar.alpha = alpha;
hmm.varpar.beta = beta;
hmm.varpar.v = v;
hmm.varpar.m = m;
hmm.varpar.W = W;
  