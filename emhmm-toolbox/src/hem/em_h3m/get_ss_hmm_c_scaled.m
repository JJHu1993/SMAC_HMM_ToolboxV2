function [ll gamma_start epsilon_sum ...
         gamma_eta_sum gamma_eta_y_sum gamma_eta_y2_sum] =  get_ss_hmm_c_scaled(varargin)
% [] =  get_ss_hmm(prior,A,B,obs)
% 
% gets sufficient statistics
% 
% input:
% prior is a N x 1 vector
% A is a N x N transition matrix
% B is a N x M observation matrix
% obs is a series of T observations
% 
% output:
% ll is the log-likelihood log P(obs)
% gamma_start is a N x 1 vector: expected starts in 1;
% epsilon_sum is a N x N matrix: expected number of trasients from u to v
% gamma_sum   is a N x M matrix: expected number of time u outputs k
%
% coded by Emanuele Coviello (11/24/2010)
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

if nargin == 2
    prior = varargin{1}.prior;
    A     = varargin{1}.A;
    emit     = varargin{1}.emit;
    obs     = varargin{2};
elseif nargin == 4
    prior = varargin{1};
    A     = varargin{2};
    emit     = varargin{3};
    obs     = varargin{4};
else
    error('Input sould be an hmm\n')
end

[T dim] = size(obs);
[N] = size(A,1);

% create probability of each observation in all the states
M = emit{1}.ncentres;
B = zeros(N,T);
eta = zeros(N,T,M);
for n = 1 : N
    [probs eta(n,:,:)] = gmmprob(emit{n}, obs);
    probs(isnan(probs)) = 0;
    B(n,:) = probs(:)'; % make it a row
end

% [alpha ll] =  fwd_hmm_c_scaled(prior,A,emit,obs);
[alpha ll] =  fwd_hmm_c_scaled(prior,A,B);
% beta =  bck_hmm_scaled(A,emit,obs);
beta =  bck_hmm_c_scaled(A,B);


% compute epsilon

epsilon = zeros(N,N,T-1);

e = ones(N,1);

for t = 1 : T-1
%     epsilonFoo = (alpha(:,t) * beta(:,t)') .* A;
%     epsilonFoo = ((alpha(:,t) * beta(:,t)') .* A) * diag(B(:,obs(t)));
    epsilonFoo = ((alpha(:,t) * beta(:,t+1)') .* A) * diag(B(:,t+1));
    
    % normalize
    epsilon(:,:,t) = epsilonFoo / (e' * epsilonFoo * e);
    % % or normalize like this
    % epsilon(:,:,t) = epsilon / l;
end

% compute gamma

gamma = zeros(N,T);

for t = 1 : T-1
    gamma(:,t) = epsilon(:,:,t) * e;
end
gamma(:,T) = alpha(:,T).*beta(:,T);
gamma(:,T) = gamma(:,T) / sum(gamma(:,T));

% compute the sufficient stats:

% to estimate initial state probabilities
gamma_start = gamma(:,1);

% to estimate transition probabilities
% epsilon_sum = sum(epsilon(:,:,1:end-1),3);
epsilon_sum = sum(epsilon,3);

% % maybe not necessary
% gamma_sum_trans   = sum(gamma(:,1:end-1),2);

% to estimate observation probabilities

gamma_eta_sum = zeros(N,M);
gamma_eta_y_sum = zeros(N,dim,M);

switch emit{1}.covar_type
    case 'diag'
        gamma_eta_y2_sum = zeros(N,dim,M);
end

% compute nu
for m = 1 : M
    eta_m = squeeze(eta(:,:,m)); % N by T
    
    % tic
        gamma_eta_sum(:,m) = sum(gamma .* eta_m,2); 
    % toc
    % tic
    %     foo_gamma_eta_sum = diag(gamma * eta_m');
    % toc
    
    gamma_eta_y_sum(:,:,m) = (gamma .* eta_m) * obs; 
    
    switch emit{1}.covar_type
        case 'diag'
            gamma_eta_y2_sum(:,:,m) = (gamma .* eta_m) * (obs.^2); 
    end
    
    
    
end

