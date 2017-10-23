function [LLG_elbo sum_w_pr sum_w_mu sum_w_Mu] = g3m_stats(g3m_b,g3m_r)
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD


% first, get the elbo of the E_gauss(b) [log p (y | gauss (r))]  (different b on different rows)
% and the sufficient statistics for the later updates
% sum_w_pr is a N by 1 cell of N by M
% sum_w_mu is a N by 1 cell of N by dim by M
% sum_w_Mu is a N by 1 cell of N by dim by M (for the diagonal case)

% number of states, e.g., how many GMM emission pdf
N = length(g3m_b);
N2 = length(g3m_r);
M = g3m_b{1}.ncentres;
dim = g3m_b{1}.nin;


LLG_elbo = zeros (N,N2);



% compute variational elbo to E_M(b),beta [log p(y | M(r),rho)] 
% and variational posteriors
sum_w_pr = cell(1,N2);
sum_w_mu = cell(1,N2);
sum_w_Mu = cell(1,N2);
    
for rho = 1 : N2

    
    foo_sum_w_pr = zeros(N,M);
    foo_sum_w_mu = zeros(N,dim,M);
    foo_sum_w_Mu = zeros(N,dim,M);
    
        
    gmmR = g3m_r{rho};
        
    for beta = 1 : N
    
        gmmB = g3m_b{beta};
        
        % compute the expected log-likelihood between the Gaussian components
        % i.e., E_M(b),beta,m [log p(y | M(r),rho,l)], for m and l 1 ...M
        [ELLs] =  compute_exp_lls(gmmB,gmmR);
        
        % compute log(omega_r) + E_M(b),beta,m [log p(y | M(r),rho,l)]
        log_theta = ELLs + ones(M,1) * log(gmmR.priors);
        
        % compute log Sum_b omega_b exp(-D(fa,gb))
        log_sum_theta = logtrick(log_theta')';
        
        % compute L_variational(M(b)_i,M(r)_j) = Sum_a pi_a [  log (Sum_b omega_b exp(-D(fa,gb)))]

         LLG_elbo(beta,rho) = gmmB.priors * log_sum_theta;
         
         % cache theta
         theta = exp(log_theta -  log_sum_theta * ones(1,M));
%          Theta{beta,rho} = theta; % indexed by m and l, [M by M]
         
         % aggregate in the output ss
         
         foo_sum_w_pr(beta,:) = gmmB.priors * theta;
         
%          for l = 1 :  M
%              
%              foo_sum_w_mu(beta,:,l) = foo_sum_w_mu(beta,:,l) + (gmmB.priors .* theta(:,l)') * gmmB.centres;
%              
%          end
         
         foo_sum_w_mu(beta,:,:) =  ( ((ones(M,1) * gmmB.priors) .* theta') * gmmB.centres )';
         
         foo_sum_w_Mu(beta,:,:) =  ( ((ones(M,1) * gmmB.priors) .* theta') * (gmmB.centres.^2 + gmmB.covars) )';
         
        
    end
    
    sum_w_pr{rho} = foo_sum_w_pr;
    sum_w_mu{rho} = foo_sum_w_mu;
    sum_w_Mu{rho} = foo_sum_w_Mu;
end

end

function [ELLs] = compute_exp_lls(gmmA,gmmB)

    dim = gmmA.nin;
    A = gmmA.ncentres;
    B = gmmB.ncentres;
    
    ELLs = zeros(A,B);
    
    for a = 1 : A
        for b = 1 : B
            
            switch(gmmA.covar_type)
%                 case 'spherical'
%                     KLs(a,b) = .5 * ( ...
%                          dim*log( gmmB.covars(1,b) / gmmA.covars(1,a) )...
%                          +  dim* gmmA.covars(1,a)/gmmB.covars(1,b) - dim ...
%                          + sum(((gmmA.centres(a,:) - gmmB.centres(b,:)).^2) ./ gmmB.covars(1,b)) ...
%                         );
                case 'diag'
%                     KLs(a,b) = .5 * ( ...
%                          sum( log(gmmB.covars(b,:)),2) - sum(log(gmmA.covars(a,:)),2 )...
%                          +  sum(gmmA.covars(a,:)./gmmB.covars(b,:),2) - dim ...
%                          + sum(((gmmA.centres(a,:) - gmmB.centres(b,:)).^2) ./ gmmB.covars(b,:),2) ...
%                         );
                      ELLs(a,b) = -.5 * ( ...
                         dim*log(2*pi) + sum( log(gmmB.covars(b,:)),2) ...
                         +  sum(gmmA.covars(a,:)./gmmB.covars(b,:),2)  ...
                         + sum(((gmmA.centres(a,:) - gmmB.centres(b,:)).^2) ./ gmmB.covars(b,:),2) ...
                        );
                    
%                 case 'full'

                    
                    
%                     inv_covB = inv(gmmB.covars(:,:,b));
%                     
%                     KLs(a,b) = .5 * ( ...
%                         log(det(gmmB.covars(:,:,b))) - log(det(gmmA.covars(:,:,a)))...
%                         + trace( inv_covB * gmmA.covars(:,:,a) ) - dim ...
%                         + (gmmA.centres(a,:) - gmmB.centres(b,:)) * inv_covB * (gmmA.centres(a,:) - gmmB.centres(b,:))' ...
%                         );
                    
                otherwise
                    error('Covarance type not supported')
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