function [alpha ll log_alpha ll_log] =  fwd_hmm_c_scaled(varargin)
% [alpha l] =  fwd_hmm(prior,A,B,obs)
% 
% does forward algorithm
% 
% input:
% prior is a N x 1 vector
% A is a N x N transition matrix
% B is a N x M observation matrix
% obs is a series of T observations
% 
% output:
% alpha is a N x T matrix with alpha(u,t) = P(obs(1:t),q_t = u)
% l is teh likelihood P(obs(1:T))
%
% coded by Emanuele Coviello (11/24/2010)
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

if nargin == 3
    prior = varargin{1};
    A     = varargin{2};
    B     = varargin{3};
    log_B     = log(B);
    T = size(B,2);

elseif nargin == 4
    prior = varargin{1};
    A     = varargin{2};
    emit     = varargin{3};
    obs     = varargin{4};
    % create probability of each observation in all the states
    M = emit{1}.ncentres;
    N = size(A,1);
    T = size(obs,1);
    B = zeros(N,T);
    log_B = zeros(N,T);
    for n = 1 : N
        probs = gmmprob(emit{n}, obs);
        probs(isnan(probs)) = 0;
%         if any(~isfinite(probs(:)))
%             pippo = 0;
%         end
%         if any(~isreal(probs(:)));
%             pippo= 0;
%         end
        B(n,:) = probs(:)'; % make it a row
        probs = gmmprob_bis(emit{n}, obs);
        log_B(n,:) = probs(:)';
        
    end
else
    error('Input sould be an hmm\n')
end


[N] = size(A,1);

alpha = zeros(N,T);
log_alpha = zeros(N,T);

% initialilze

alpha(:,1) = prior .* B(:,1);
log_alpha(:,1) = log(prior) + log_B(:,1);

% scale
foo = sum(alpha(:,1));
alpha(:,1) = alpha(:,1) / foo;
scale(1) = foo;

% loop

for t = 1 : T - 1
    alpha(:,t+1) = (A' * alpha(:,t)) .* B(:,t+1);
    
    log_alpha(:,t+1) = logtrick(log(A) + log_alpha(:,t) * ones(1,N))' + log_B(:,t+1);
    %scale
    foo = sum(alpha(:,t+1));
    alpha(:,t+1) = alpha(:,t+1) / foo;
    scale(t+1) = foo;
end


% if any(isnan(scale)) 
%     pollo = 0;
% end
   
ll_log = logtrick(log_alpha(:,end));

% compute log-likelihood
if any(scale==0)
   ll = -inf;
elseif any(~isfinite(scale))  % or ~isreal?
    ll = -inf;
else
   ll = sum(log(scale));
end