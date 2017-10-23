function [beta log_beta] =  bck_hmm_c_scaled(A,emit,obs)
% [beta] =  fwd_hmm(A,B,obs)
% 
% does backward algorithm
% 
% input:
% A is a N x N transition matrix
% B is a N x M observation matrix
% obs is a series of T observations
% 
% output:
% beta is a N x T matrix with alpha(u,t) = P(obs(1:t),q_t = u)
%
% coded by Emanuele Coviello (11/24/2010)
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

if ~exist('obs','var')
    B = emit;
    T = size(B,2);
    log_B     = log(B);
else
    % create probability of each observation in all the states
    T = size(obs,1);
    M = emit{1}.ncentres;
    B = zeros(N,T);
    log_B = zeros(N,T);
    N = size(A,1);
    for n = 1 : N
        probs = gmmprob(emit{n}, obs);
        B(n,:) = probs(:)'; % make it a row
        probs = gmmprob_bis(emit{n}, obs);
        log_B(n,:) = probs(:)';
    end
end


[N ] = size(A,1);



beta = zeros(N,T);
log_beta = zeros(N,T);

% initialilze

beta(:,T) = ones(N,1);

log_beta(:,T) = zeros(N,1);

foo = sum(beta(:,T));
beta(:,T) = beta(:,T)/foo;

% loop

for t = T - 1 : -1 : 1
%     beta(:,t) = (A * B(:,obs(t+1))) .* beta(:,t+1);
    beta(:,t) = A * ( B(:,t+1) .* beta(:,t+1));
    log_beta(:,t) = logtrick(log(A') + ( log_B(:,t+1) + beta(:,t+1)) * ones(1,N))';
    foo = sum(beta(:,t));
    beta(:,t) = beta(:,t)/foo;
    
end

% T = length(obs);
% [N M] = size(B);
% 
% beta = zeros(N,T);
% 
% % initialilze
% 
% beta(:,T) = ones(N,1);
% 
% foo = sum(beta(:,T));
% beta(:,T) = beta(:,T)/foo;
% 
% % loop
% 
% for t = T - 1 : -1 : 1
% %     beta(:,t) = (A * B(:,obs(t+1))) .* beta(:,t+1);
%     beta(:,t) = A * ( B(:,obs(t+1)) .* beta(:,t+1));
%     foo = sum(beta(:,t));
%     beta(:,t) = beta(:,t)/foo;
%     
% end
% 
