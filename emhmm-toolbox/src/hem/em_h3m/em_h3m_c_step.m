function h3m = em_h3m_c_step(h3m,O,mopt)
% O = {obs1 obs2 obs3 ... obsI}
% h3m.K
% h3m.hmm = {hmm1 hmm2 ... hmmK}
% h3m.omega = [omega1 omega2 ... omegaK]
% hmm1.A hmm1.B hmm1.prior
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD


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


if mopt.story
    story = {};
    time = [];
end

here_time = tic;
if isfield(mopt,'start_time')
    a_time = mopt.start_time;
else
    a_time = here_time;
end


    if mopt.story
        story{end+1} = h3m;
        time(end+1) = toc(a_time);
    end

num_iter = 0;

N = size(h3m.hmm{1}.A,1);
M = (h3m.hmm{1}.emit{1}.ncentres);
I = length(O);
K = h3m.K;

for j = 1 : K
    for n = 1 : N
        h3m.hmm{j}.emit{n}.covars = h3m.hmm{j}.emit{n}.covars + reg_cov;
    end
end
    

[T dim] = size(O{1});

h3m.LogLs = [];

inf_norm = 1;
if isfield(mopt,'inf_norm')
    switch mopt.inf_norm
        case 't'
            inf_norm = T;
    end
end

while 1

    %%%%%%%%%%%%%%%%%%%%
    %%%    E-step    %%%
    %%%%%%%%%%%%%%%%%%%%

    % compute soft assignments

    
    
%     [ll z_hat] = emh3m_inference_scaled(h3m,O);

    [llt z_hat] = h3m_c_inference(h3m,O,inf_norm);
    z_hat(isnan(z_hat)) = 0;
    
    ll = z_hat .* llt;
    % set to zero the -inf * 0 terms
    ll(intersect(find(isinf(llt)) , find((z_hat ==0)))) = 0;

    
%     h3m.ll = ll;
%     h3m.z = z_hat;
    
    new_LogLikelihood = ones(1,I) * ll * ones(K,1);
    
    old_LogLikelihood = h3m.LogL;
    h3m.LogL = new_LogLikelihood;
    h3m.LogLs(end+1) = new_LogLikelihood;
    h3m.Z = z_hat;
    
   if isnan(new_LogLikelihood)
% % %         fprintf('restarting estimation ...\n')
        break
    end

    
    % check whether to continue or to return
    
    
    stop = 0;
    if num_iter > 1
        changeLL = (new_LogLikelihood - old_LogLikelihood) / abs(old_LogLikelihood);
    else
        changeLL = inf;
    end
    
    if changeLL<0
% % %         fprintf('The change in log likelihood is negative!!!\n')
    end
    
    
    switch mopt.termmode
        case 'L'
            if (abs(changeLL) < mopt.termvalue) 
                stop = 1;
            end
        case 'T'
            time_elapsed4now = toc(here_time);
            if (time_elapsed4now >= mopt.termvalue) 
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

    %%%%%%%%%%%%%%%%%%%%
    %%%    M-step    %%%
    %%%%%%%%%%%%%%%%%%%%

    % compute new parameters

    h3m_new = h3m;

    % new weights (omega)
    omega_new = (ones(1,I) / I) * z_hat;
    h3m_new.omega = omega_new;


    % loop all the hmm components of the new mixture
    for j = 1 : K

        new_prior = zeros(N,1);
        new_A     = zeros(N,N);
        new_mix = h3m.hmm{j}.emit;
        outer = cell(1,N);
        % put to zero all fields
        for n = 1 : N
            new_mix{n}.priors  = new_mix{n}.priors  *0;
            new_mix{n}.centres = new_mix{n}.centres *0;
            outer{n} = new_mix{n}.covars  *0;
        end
        
        for i = 1 : I
            [foo gamma_start epsilon_sum gamma_eta_sum gamma_eta_y_sum gamma_eta_y2_sum] ...
                =  get_ss_hmm_c_scaled(h3m.hmm{j},O{i});
            
            
            % check this                                           <<<-----
%             l(isnan(l)) = 0;
            gamma_start(isnan(gamma_start)) = 0;
            epsilon_sum(isnan(epsilon_sum)) = 0;
            gamma_eta_sum(isnan(gamma_eta_sum)) = 0;
            gamma_eta_y_sum(isnan(gamma_eta_y_sum)) = 0;
            gamma_eta_y2_sum(isnan(gamma_eta_y2_sum)) = 0;
            
            if z_hat(i,j) > 0

                new_prior = new_prior + z_hat(i,j) * gamma_start;

                new_A     = new_A     + z_hat(i,j) * epsilon_sum;

                for n = 1 : N
                    new_mix{n}.priors  = new_mix{n}.priors  + z_hat(i,j) * gamma_eta_sum(n,:);
    % % %                 new_mix{n}.centres = new_mix{n}.centres + z_hat(i,j) * squeeze(gamma_eta_y_sum(n,:,:))';
    %                 foo = 
                    new_mix{n}.centres = new_mix{n}.centres + z_hat(i,j) * reshape(gamma_eta_y_sum(n,:,:),dim,[])';
    %                 new_mix{n}.covars  = new_mix{n}.covars  + z_hat(i,j) * gamma_eta_y2_sum;
                    % this works only for diagonal covariance ...
    % % %                 outer{n}  = outer{n}  + z_hat(i,j) * squeeze(gamma_eta_y2_sum(n,:,:))';
                    outer{n}  = outer{n}  + z_hat(i,j) * reshape(gamma_eta_y2_sum(n,:,:),dim,[])';
                end
                
            else
                pippo = 0;
            end
            
        end

        % normalize things
        new_prior = new_prior / sum(new_prior);
        new_A     = new_A    ./ repmat(sum(new_A,2),1,N);
        for n = 1 : N
            b4_norm_centres = new_mix{n}.centres;
            new_mix{n}.centres = b4_norm_centres ./ (new_mix{n}.priors' * ones(1,dim));
            % this works only for diagonal covariance ...
            Sigma  = outer{n}  - 2* (b4_norm_centres.*new_mix{n}.centres) + (new_mix{n}.centres.^2) .* (new_mix{n}.priors' * ones(1,dim));
            Sigma = Sigma ./ (new_mix{n}.priors' * ones(1,dim));
            new_mix{n}.covars = Sigma + reg_cov;
            new_mix{n}.priors  = new_mix{n}.priors  ./ sum(new_mix{n}.priors);
        end


        h3m_new.hmm{j}.prior = new_prior;
        h3m_new.hmm{j}.A     = new_A;
        h3m_new.hmm{j}.emit  = new_mix;
    end

%     h3m_new = h3m_c_regularize(h3m_new,0.00001);
    
    h3m = h3m_new;
    
    h3m = regularize_emit(h3m);
    
    if mopt.story
        story{end+1} = h3m;
        time(end+1) = toc(a_time);
    end
    
end

h3m.elapsed_time = toc(here_time);
if mopt.story
h3m.story = story;
h3m.time = time;
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h3m = regularize_emit(h3m,min_cov)
    
    N = size(h3m.hmm{1}.A,1);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% if a component gets 0 prior ...
% ind_zero = find(h3m.omega <= (1 /(h3m.K * 1000)));
ind_zero = find(h3m.omega <= 0);
    for i_z = ind_zero;
        fprintf('!!! modifying h3m: one hmm has zero prior \n')
        [foo highest] = max(h3m.omega);
        h3m.omega([i_z highest]) = h3m.omega(highest)/2;
        % renormalize for safety
        h3m.omega = h3m.omega / sum(h3m.omega);
        h3m.hmm{i_z} = h3m.hmm{highest};
        % perturb
        h3m.hmm{i_z}.prior = h3m.hmm{highest}.prior + (.1/N) * rand(size(h3m.hmm{highest}.prior));
        A = h3m.hmm{highest}.A;
        f_zeros = find(A == 0);
        A     = A     + (.1/N) * rand(size(A));
        A(f_zeros) = 0;
        
        h3m.hmm{i_z}.A = A;
        % renormalize
        h3m.hmm{i_z}.prior = h3m.hmm{i_z}.prior / sum(h3m.hmm{i_z}.prior);
        h3m.hmm{i_z}.A     = h3m.hmm{i_z}.A    ./   repmat(sum(h3m.hmm{i_z}.A,2),1,N);
    end


for j = 1 :  h3m.K
    
    % if the prior is to small, eliminate 
    min_prior_tol = 0; % = 1 / (100 * length(h3m.hmm{j}.emit{1}.priors));
    
    
    for n = 1 : length(h3m.hmm{j}.emit)
        
        
        %%%%%%% if some of the emission prob is lower than the tolerance, replace it ...
        ind_zero = find(h3m.hmm{j}.emit{n}.priors <= min_prior_tol);
        for i_z = ind_zero;
            fprintf('!!! modifying gmm emission: one component is lower than tolerance \n')
            [foo highest] = max(h3m.hmm{j}.emit{n}.priors);
            h3m.hmm{j}.emit{n}.priors([i_z highest]) = h3m.hmm{j}.emit{n}.priors(highest)/2;
            % renormalize for safety
            h3m.hmm{j}.emit{n}.priors = h3m.hmm{j}.emit{n}.priors / sum(h3m.hmm{j}.emit{n}.priors);

            h3m.hmm{j}.emit{n}.centres(i_z,:) = h3m.hmm{j}.emit{n}.centres(highest,:);
            h3m.hmm{j}.emit{n}.covars(i_z,:)  = h3m.hmm{j}.emit{n}.covars(highest,:);

            % perturb only the centres
             h3m.hmm{j}.emit{n}.centres(i_z,:) =  h3m.hmm{j}.emit{n}.centres(i_z,:) + (0.01 * rand(size(h3m.hmm{j}.emit{n}.centres(i_z,:)))) ...
                 .* h3m.hmm{j}.emit{n}.centres(i_z,:);
%              h3m.hmm{j}.emit{n}.centres(i_z,:) =  h3m.hmm{j}.emit{n}.centres(i_z,:) + 0 ...
%                  .* h3m.hmm{j}.emit{n}.centres(i_z,:);
        end
        
        
        
        % regularize the covariance matrices
        covars = h3m.hmm{j}.emit{n}.covars;
        if any(covars(:) == 0)
            for f = 1 : size(covars,2)
                cov = covars(:,f);
                % find the minimun greater than zero
                min_g0 = min(setdiff(cov,0));
                if isempty(min_g0)
                    min_g0 = min(setdiff(covars(:),0));
                end
                if isempty(min_g0)
                    min_g0 = 0.00001;
                end
                cov(find(cov==0)) = min_g0;
                covars(:,f) = cov;
            end
            h3m.hmm{j}.emit{n}.covars = covars;
        end
    end
end



end