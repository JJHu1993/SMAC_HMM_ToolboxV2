function [LL_elbo ...
                sum_nu_1 ...
                update_emit_pr ...
                update_emit_mu ...
                update_emit_Mu  ...
                sum_xi ...
                sum_t_nu] = hem_hmm_bwd_fwd(hmm_b,hmm_r,T,smooth)
% run the Hierarchical bwd and fwd recursions. This will compute:
% the ELBO on E_M(b)i [log P(y_1:T | M(r)i)]
% the nu_1^{i,j} (rho)
% the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu..
% the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. mu ..
% the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. M .. which accounts for the mu mu' and the sigma
% [the last 3 are NOT normalized YET, of course]
% the sum_t xi_t^{i,j} (rho,sigma)     (t = 2 ... T)
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD


if isempty(smooth)
    smooth = 1;
end


% number of states 
N = size(hmm_b.A,1);
N2 = size(hmm_r.A,1);
% number of mixture components in each state emission probability
M = hmm_b.emit{1}.ncentres;
% dimension of the emission variable
dim = hmm_b.emit{1}.nin;

% first, get the elbo of the E_gauss(b) [log p (y | gauss (r))]  (different b on different rows)
% and the sufficient statistics for the later updates
% sum_w_pr is a N by 1 cell of N by M
% sum_w_mu is a N by 1 cell of N by dim by M
% sum_w_Mu is a N by 1 cell of N by dim by M (for the diagonal case)
[LLG_elbo sum_w_pr sum_w_mu sum_w_Mu] = g3m_stats(hmm_b.emit,hmm_r.emit);

LLG_elbo = LLG_elbo / smooth; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% do the backward recursion %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ab = hmm_b.A;
Ar = hmm_r.A;

% allocate Theta, i.e., the VA parameters in the form of a CMC

Theta = zeros(N2,N2,N,T); % the dimensions refer to rho sigma (states of M(r)) gamma (state of M(b)) and time
        % rho sigma gamma t

% allocate the log-likelihood

LL_old = zeros(N,N2);       % the dimensions refer to gamma (state of M(b)) and sigma (states of M(r)) 

% the log-likelihood can be intialized by all zeros ...



for t = T : -1 : 2
    
    LL_new = zeros(size(LL_old));

    for rho = 1 : N2

        logtheta = (log(Ar(rho,:))' * ones(1,N)) + LLG_elbo' + LL_old';
% % % logtheta = (log(Ar(rho,:))' * ones(1,N)) + LLG_elbo + LL_old;

        logsumtheta = logtrick(logtheta);

        LL_new(:,rho) =  Ab * logsumtheta';

        % normalize so that each clmn sums to 1 (may be not necessary ...) 
        theta = exp(logtheta - ones(N2,1) * logsumtheta);

        % and store for later

        Theta(rho,:,:,t) = theta;

    end
    
    LL_old = LL_new;

end

% terminate the recursion

logtheta = (log(hmm_r.prior) * ones(1,N)) + LLG_elbo' + LL_old';
% % % logtheta = (log(hmm_r.prior) * ones(1,N)) + LLG_elbo + LL_old;

logsumtheta = logtrick(logtheta);

LL_elbo =  hmm_b.prior' * logsumtheta';

% normalize so that each clmn sums to 1 (may be not necessary ...) 
theta = exp(logtheta - ones(N2,1) * logsumtheta);

% and store for later
Theta_1 = theta;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% do the forward recursion  %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% rather then saving all intermediate values, just cache the cumulative
% values that are needed for the updates (it saves a lot of memory)

% nu is N by N, first dimension indexed by sigma (M(r)), second by gamma (M(b))

% initialize
nu = (ones(N2,1) * hmm_b.prior') .* Theta_1; % N by N (indexed by sigma and gamma)



% CACHE: sum_gamma nu_1(sigma,gamma) (this is one of the outputs ...)
sum_nu_1 = sum(nu,2)';

% CACHE: sum_t nu_t(sigma, gamma)
% sum_t_nu = zeros(N,N);
sum_t_nu = nu;

% CACHE: sum_t sum_gamma xi(rho,sigma,gamma,t)
sum_t_sum_g_xi = zeros(N2,N2); % N by N (indexed by rho and sigma)

for t = 2 : T
    
    % compute the inner part of the update of xi (does not depend on sigma)
    foo = nu * Ab; % indexed by rho gamma
    
    for sigma = 1 : N2
        
        
        % new xi
        % xi(:,sigma,:,t) = foo .* squeeze(Theta(:,sigma,:,t));
        %xi_foo = foo .* squeeze(Theta(:,sigma,:,t)); % (indexed by rho gamma);
        
        % ABC: bug fix when another dim is 1
        xi_foo = foo .* reshape(Theta(:,sigma,:,t), [size(Theta,1), size(Theta,3)]); % (indexed by rho gamma);
        
        
        
        % CACHE:
        sum_t_sum_g_xi(:,sigma) = sum_t_sum_g_xi(:,sigma) + sum(xi_foo,2);
        
        % new nu
        % nu(sigma,:) = ones(1,N) * squeeze(xi(:,sigma,:,t));
        nu(sigma,:) = ones(1,N2) * xi_foo;
        
        
    end
    
    % CACHE: in the sum_t nu_t(sigma, gamma)
    sum_t_nu = sum_t_nu + nu;
    
end

% this is one of the outputs ...
sum_xi = sum_t_sum_g_xi;


%%%% now prepare the cumulative sufficient statistics for the reestimation
%%%% of the emission distributions

update_emit_pr = zeros(N2,M);
update_emit_mu = zeros(N2,dim,M);
switch hmm_b.emit{1}.covar_type
    case 'diag'
        update_emit_Mu = zeros(N2,dim,M);
end

% loop all the emission GMM of each state

for sigma = 1 : N2
    
    update_emit_pr(sigma,:) = sum_t_nu(sigma,:) * sum_w_pr{sigma};
    
    foo_sum_w_mu = sum_w_mu{sigma};
    foo_sum_w_Mu = sum_w_Mu{sigma};
    
    for l = 1 : M
        
        update_emit_mu(sigma,:,l) = sum_t_nu(sigma,:) * foo_sum_w_mu(:,:,l);
        
        switch hmm_b.emit{1}.covar_type
            case 'diag'
                update_emit_Mu(sigma,:,l) = sum_t_nu(sigma,:) * foo_sum_w_Mu(:,:,l);
        end
        
        
    end
    
    
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [s] = logtrick(lA)
% logtrick - "log sum trick" - calculate log(sum(A)) using only log(A) 
%
%   s = logtrick(lA)
%
%   lA = column vector of log values
%
%   if lA is a matrix, then the log sum is calculated over each column
% 

[mv, mi] = max(lA, [], 1);
temp = lA - repmat(mv, size(lA,1), 1);
cterm = sum(exp(temp),1);
s = mv + log(cterm);
end



 