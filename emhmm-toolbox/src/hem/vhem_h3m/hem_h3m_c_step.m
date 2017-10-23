function h3m_r = hem_h3m_c_step(h3m_r,h3m_b,mopt) 
% h3m_r = initialization for the reduced mixture output
% h3m_b = base mixture input
% mopt  = options 
%       .K     = number of mixtures in the reduced model
%       .Nv     = number of virtual samples
%       .tau   = length of virtual sequences
%       .termmode  = how to terminate EM
%       .termvalue = when to terminate the EM
% h3m.K
% h3m.hmm = {hmm1 hmm2 ... hmmK}
% h3m.omega = [omega1 omega2 ... omegaK]
% hmm1.A hmm1.emit hmm1.prior
%
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

% 2016-12-09: ABC - added output for emission populatino size



tic;


h3m_r.LogLs = [];

num_iter = 0;

% number of components in the base and reduced mixtures
Kb = h3m_b.K;
Kr = h3m_r.K;

% number of states 
N = size(h3m_b.hmm{1}.A,1);
Nr = size(h3m_r.hmm{1}.A,1);
% number of mixture components in each state emission probability
M = h3m_b.hmm{1}.emit{1}.ncentres;
% length of the virtual sample sequences
T = mopt.tau;
% dimension of the emission variable
dim = h3m_b.hmm{1}.emit{1}.nin;
% number of virtual samples 
virtualSamples = mopt.Nv;
% correct
N_i = virtualSamples * h3m_b.omega * Kb; N_i = N_i(:);

if isfield(mopt,'reg_cov')
    reg_cov = mopt.reg_cov;
else
    reg_cov = 0;
end

if ~isfield(mopt,'min_iter')
    mopt.min_iter = 0;
end

if ~isfield(mopt,'story')
    mopt.story = 0;
end

here_time = tic;
if isfield(mopt,'start_time')
    a_time = mopt.start_time;
else
    a_time = here_time;
end

if mopt.story
    story = {};
    time = [];
end

if mopt.story
    story{end+1} = h3m_r;
    time(end+1) = toc(a_time);
end


for j = 1 : Kr
    for n = 1 : Nr  % bug fix (was N before)
        h3m_r.hmm{j}.emit{n}.covars = h3m_r.hmm{j}.emit{n}.covars + reg_cov;
    end
end

switch mopt.inf_norm
    case ''
        inf_norm = 1;
    case 'n'
        inf_norm = virtualSamples / Kb;
    case {'tn' 'nt'}
        inf_norm = T * virtualSamples / Kb;
    case {'t'}
        inf_norm = T;
    
end

smooth =  mopt.smooth;
    

% start looping variational E step and M step

L_elbo           = zeros(Kb,Kr);
nu_1             = cell(Kb,Kr);
update_emit_pr   = cell(Kb,Kr);
update_emit_mu   = cell(Kb,Kr);
update_emit_M    = cell(Kb,Kr);
sum_xi           = cell(Kb,Kr);

while 1

    %%%%%%%%%%%%%%%%%%%%
    %%%    E-step    %%%
    %%%%%%%%%%%%%%%%%%%%
    
    % loop reduced mixture components
    for j = 1 : Kr
        
        hmm_r = h3m_r.hmm{j};
        
        % loop base mixture components
        for i = 1 : Kb
            
            hmm_b = h3m_b.hmm{i};
            
            % run the Hierarchical bwd and fwd recursions. This will compute:
            % the ELBO on E_M(b)i [log P(y_1:T | M(r)i)]
            % the nu_1^{i,j} (rho)
            % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu..
            % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. mu ..
            % the sum_gamma  [sum_t nu_t^{i,j} (rho,gamma)] sum_m c.. nu.. M .. which accounts for the mu mu' and the sigma
            % [the last 3 are not normalized, of course]
            % the sum_t xi_t^{i,j} (rho,sigma)     (t = 2 ... T)
            [L_elbo(i,j) ...
                nu_1{i,j} ...
                update_emit_pr{i,j} ...
                update_emit_mu{i,j} ...
                update_emit_M{i,j}  ...
                sum_xi{i,j} ...
                ] = hem_hmm_bwd_fwd(hmm_b,hmm_r,T,smooth);
            
        % end loop base     
        end
        
    % end loop reduced    
    end
    
    % compute the z_ij
    % this is not normalized ...
    % log_Z = ones(Kb,1) * log(h3m_r.omega) + diag(N_i) * L_elbo; 
    L_elbo = L_elbo / (inf_norm);
    log_Z = ones(Kb,1) * log(h3m_r.omega) + (N_i(:) * ones(1,Kr)) .* L_elbo;
    Z = exp(log_Z - logtrick(log_Z')' * ones(1,Kr));
    
    % compute the elbo to total likelihood ...
    

%     ll = Z .* (log_Z - log(Z));
%     
%     ll(isnan(ll)) = 0;
%      
%     
%     % new_LogLikelihood = sum(sum(ll));
%     new_LogLikelihood_foo = ones(1,Kb) * ll * ones(Kr,1);
    % these should be the same
    new_LogLikelihood = sum(logtrick(log_Z')');
    
    % update the log likelihood in the reduced mixture
    old_LogLikelihood = h3m_r.LogL;
    h3m_r.LogL = new_LogLikelihood;
    h3m_r.LogLs(end+1) = new_LogLikelihood;
    h3m_r.Z = Z; 
    
    
    
    % check whether to continue with a new iteration or to return
    
    
    stop = 0;
    if num_iter > 1
        changeLL = (new_LogLikelihood - old_LogLikelihood) / abs(old_LogLikelihood);
    else
        changeLL = inf;
    end
    
    if changeLL<0
% % %         fprintf('\nThe change in log likelihood is negative!!!')
    end
    
    
    switch mopt.termmode
        case 'L'
            if (changeLL < mopt.termvalue) 
                stop = 1;
            end
    end
    
    if (num_iter > mopt.max_iter)
        stop = 1;
    end
        
    if (stop) && (num_iter >= mopt.min_iter)
        break
    end
    
    num_iter = num_iter + 1;
    
    old_LogLikelihood = new_LogLikelihood;
    
    %%%%%%%%%%%%%%%%%%%%
    %%%    M-step    %%%
    %%%%%%%%%%%%%%%%%%%%

    % compute new parameters

    h3m_r_new = h3m_r;   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%  re-estimation of the component weights (omega)
    
    omega_new = (ones(1,Kb) / Kb) * Z;
    h3m_r_new.omega = omega_new;
    
    
    % scale the z_ij by the number of virtual samples
    Z = Z .* (N_i(:) * ones(1,Kr));
    N2 = size(h3m_r.hmm{j}.A,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% re- estimation of the HMMs parameters
    
    % look at one mixture component at a time
    for j = 1 : Kr
        
        new_prior = zeros(N2,1);
        new_A     = zeros(N2,N2);
        
        new_Gweight = cell(1,N2);
        new_Gmu     = cell(1,N2);
        new_GMu     = cell(1,N2);
        
        for n = 1 : N2
            new_Gweight{n}  = zeros(1,M);
            new_Gmu{n}      = zeros(M,dim);
            
            switch mopt.emit.covar_type
                case 'diag'
                new_GMu{n}       = zeros(M,dim);
            end
        end
        
        % loop all the components of the base mixture
        for i = 1 : Kb
            
           
            if Z(i,j) > 0
            
                nu = nu_1{i,j};              % this is a 1 by N vector
                xi = sum_xi{i,j};            % this is a N by N matrix (from - to)
                up_pr = update_emit_pr{i,j}; % this is a N by M matrix
                up_mu = update_emit_mu{i,j}; % this is a N by dim by M matrix
                up_Mu  = update_emit_M{i,j}; % this is a N by dim by M matrix

                new_prior = new_prior + Z(i,j) * nu';
                new_A     = new_A     + Z(i,j) * xi;

                for n = 1 : N2
                    new_Gweight{n}  = new_Gweight{n} + Z(i,j) * up_pr(n,:);
    % % %                 new_Gmu{n}      = new_Gmu{n}     + Z(i,j) * squeeze(up_mu(n,:,:))';
                    new_Gmu{n}      = new_Gmu{n}     + Z(i,j) * reshape(up_mu(n,:,:),dim,[])';

                    switch mopt.emit.covar_type
                        case 'diag'
    % % %                     new_GMu{n}  = new_GMu{n}     + Z(i,j) * squeeze(up_Mu(n,:,:))';
                        new_GMu{n}  = new_GMu{n}     + Z(i,j) * reshape(up_Mu(n,:,:),dim,[])';
                    end
                end
            end
            
            
            
        end
        
        % normalize things, i.e., divide by the denominator
        h3m_r_new.hmm{j}.prior = new_prior / sum(new_prior);
        h3m_r_new.hmm{j}.A     = new_A    ./ repmat(sum(new_A,2),1,N2);
        
        % ABC 2016-12-09 - save the counts in each emission
        h3m_r_new.hmm{j}.counts_emit = sum(new_A,1) + new_prior';
       
        % normalize the emission distrbutions
        for n = 1 : N2
            
            %normalize the mean
            h3m_r_new.hmm{j}.emit{n}.centres = new_Gmu{n} ./ (new_Gweight{n}' * ones(1,dim));
            
            %normalize the covariance
            switch mopt.emit.covar_type
                case 'diag'
                    Sigma  = new_GMu{n}  - 2* (new_Gmu{n} .* h3m_r_new.hmm{j}.emit{n}.centres) ...
                        + (h3m_r_new.hmm{j}.emit{n}.centres.^2) .* (new_Gweight{n}' * ones(1,dim));
                    h3m_r_new.hmm{j}.emit{n}.covars = Sigma ./ (new_Gweight{n}' * ones(1,dim));
                    h3m_r_new.hmm{j}.emit{n}.covars =  h3m_r_new.hmm{j}.emit{n}.covars + reg_cov;
            end
            
            % normalize the mixture weight of the gaussian
            h3m_r_new.hmm{j}.emit{n}.priors = new_Gweight{n} ./ sum(new_Gweight{n});
            
% % %             %%%%%%% if some of the emission prob is zero, replace it ...
% % %             ind_zero = find(h3m_r_new.hmm{j}.emit{n}.priors == 0);
% % %             for i_z = ind_zero;
% % %                 fprintf('!!! modifying gmm emission: one component is zero \n')
% % %                 [foo highest] = max(h3m_r_new.hmm{j}.emit{n}.priors);
% % %                 h3m_r_new.hmm{j}.emit{n}.priors([i_z highest]) = h3m_r_new.hmm{j}.emit{n}.priors(highest)/2;
% % %                 % renormalize for safety
% % %                 h3m_r_new.hmm{j}.emit{n}.priors = h3m_r_new.hmm{j}.emit{n}.priors / sum(h3m_r_new.hmm{j}.emit{n}.priors);
% % %                 
% % %                 h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) = h3m_r_new.hmm{j}.emit{n}.centres(highest,:);
% % %                 h3m_r_new.hmm{j}.emit{n}.covars(i_z,:)  = h3m_r_new.hmm{j}.emit{n}.covars(highest,:);
% % %                 
% % %                 % perturb only the centres
% % %                  h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) =  h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) + (0.01 * rand(size(h3m_r_new.hmm{j}.emit{n}.centres(i_z,:))))...
% % %                      .*h3m_r_new.hmm{j}.emit{n}.centres(i_z,:);
% % %             end
             
             
        end
        
    % end loop for all mixture components of the reduced model
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% if a component gets 0 prior ...
    ind_zero = find(h3m_r_new.omega == 0);
    for i_z = ind_zero;
        fprintf('!!! modifying h3m: one hmm has zero prior \n')
        [foo highest] = max(h3m_r_new.omega);
        h3m_r_new.omega([i_z highest]) = h3m_r_new.omega(highest)/2;
        % renormalize for safety
        h3m_r_new.omega = h3m_r_new.omega / sum(h3m_r_new.omega);
        h3m_r_new.hmm{i_z} = h3m_r_new.hmm{highest};
        % perturb
        h3m_r_new.hmm{i_z}.prior = h3m_r_new.hmm{highest}.prior + (.1/N) * rand(size(h3m_r_new.hmm{highest}.prior));
        A = h3m_r_new.hmm{highest}.A;
        f_zeros = find(A == 0);
        A = (.1/N) * rand(size(A));
        A(f_zeros) = 0;
        
        h3m_r_new.hmm{i_z}.A     = A;
        % renormalize
        h3m_r_new.hmm{i_z}.prior = h3m_r_new.hmm{i_z}.prior / sum(h3m_r_new.hmm{i_z}.prior);
        h3m_r_new.hmm{i_z}.A     = h3m_r_new.hmm{i_z}.A    ./   repmat(sum(h3m_r_new.hmm{i_z}.A,2),1,N);
    end
    
    for j = 1 : Kr
        for n = 1 : N2
                %%%%%%% if some of the emission prob is zero, replace it ...
            ind_zero = find(h3m_r_new.hmm{j}.emit{n}.priors == 0);
            for i_z = ind_zero;
                fprintf('!!! modifying gmm emission: one component is zero \n')
                [foo highest] = max(h3m_r_new.hmm{j}.emit{n}.priors);
                h3m_r_new.hmm{j}.emit{n}.priors([i_z highest]) = h3m_r_new.hmm{j}.emit{n}.priors(highest)/2;
                % renormalize for safety
                h3m_r_new.hmm{j}.emit{n}.priors = h3m_r_new.hmm{j}.emit{n}.priors / sum(h3m_r_new.hmm{j}.emit{n}.priors);
                
                h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) = h3m_r_new.hmm{j}.emit{n}.centres(highest,:);
                h3m_r_new.hmm{j}.emit{n}.covars(i_z,:)  = h3m_r_new.hmm{j}.emit{n}.covars(highest,:);
                
                % perturb only the centres
                 h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) =  h3m_r_new.hmm{j}.emit{n}.centres(i_z,:) + (0.01 * rand(size(h3m_r_new.hmm{j}.emit{n}.centres(i_z,:))))...
                     .*h3m_r_new.hmm{j}.emit{n}.centres(i_z,:);
            end
        end
    end
    
    
    
    % update the model
    h3m_r = h3m_r_new;
    
    if mopt.story
        story{end+1} = h3m_r;
        time(end+1) = toc(a_time);
    end
    
    
end

h3m_r.elapsed_time = toc(here_time);
if mopt.story
h3m_r.story = story;
h3m_r.time = time;
end

% enf of the function
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
