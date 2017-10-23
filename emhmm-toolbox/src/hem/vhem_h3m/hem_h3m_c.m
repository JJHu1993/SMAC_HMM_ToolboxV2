function h3m_out = hem_h3m_c(h3m_b,mopt,emopt) 
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

% 2016-12-09: ABC - added initialization using splitting

% need some functions from the gmm package... 

VERBOSE_MODE = mopt.verbose;

if (mopt.K == length(h3m_b.omega)) || (mopt.K == 0)
    h3m_out = h3m_b;
    return
end


LL_best = -inf;

re_do = 0;

switch (mopt.initmode)
  %% ABC 2016-12-09: initialize use splitting
  case 'split_old'
    for KK = 1:mopt.K
      if KK==1
        hemopt2 = mopt;
        hemopt2.initmode = 'base';
        hemopt2.termvalue = 1e-4;
        emopt2.trials = 5;
        hemopt2.K = 1;
        tmph3m = hem_h3m_c(h3m_b,hemopt2,emopt2);                        
      else
        tmph3m_old = tmph3m;
        
        
        % find component w/ most members
        [~,mi] = max(tmph3m_old.omega);
        
        % find ROI with largest prior
        [~,ri] = max(tmph3m_old.hmm{mi}.prior);
        
        % perturb HMM
        for ri=1:length(tmph3m.hmm{mi}.prior)
          tmph3m.hmm{KK} = tmph3m.hmm{mi};
          tmph3m.hmm{KK}.emit{ri}.centres = tmph3m.hmm{KK}.emit{ri}.centres*1.1;
        end      
        
        % fix omega
        tmp = tmph3m.omega(mi)/2;
        tmph3m.omega(KK) = tmp;
        tmph3m.omega(mi) = tmp;
        tmph3m.K = KK;
        
        % use uniform distributions for transitions and prior 
        for m=1:length(tmph3m.omega)
          S = length(tmph3m.hmm{m}.prior);
          tmph3m.hmm{m}.prior = ones(S,1)/S;
          tmph3m.hmm{m}.A     = ones(S,S)/S;
        end
        
        tmph3m = hem_h3m_c_step(tmph3m,h3m_b,mopt);
      end
    end
        
    h3m_out = tmph3m;
    
    %% initialize using splitting - 
  case 'split'
    % 1. start with K=1, N=1
    % 2. split into K=2, N=1. repeat until desired K reached
    % 3. now split N=2. repeat until desired S reached
    
    % --- step 1 - learn H3M with K=1, N=1 ---
    tmph3m.K = 1;
    tmph3m.omega = 1;
    
    % make the N=1 HMM
    tmphmm.prior = 1;
    tmphmm.A     = 1;
    
    % only works for gmm emissions
    % initialize by averaging means and covariances of Gaussian emissions
    mym = 0; myc = 0; myn = 0;
    for i=1:length(h3m_b.hmm)
      for j=1:length(h3m_b.hmm{i}.emit)
        mym = mym + h3m_b.hmm{i}.emit{j}.centres;
        myc = myc + h3m_b.hmm{i}.emit{j}.covars;
        myn = myn+1;
      end
    end    
    tmphmm.emit{1}.centres = mym / myn;
    tmphmm.emit{1}.covars  = myc / myn;
    tmphmm.emit{1}.covar_type = h3m_b.hmm{1}.emit{1}.covar_type;
    tmphmm.emit{1}.ncentres = 1;
    tmphmm.emit{1}.priors = 1;
    tmphmm.emit{1}.type     = 'gmm';  % a GMM with one component
    tmphmm.emit{1}.nin      = length(mym);
            
    tmph3m.hmm{1} = tmphmm;
    tmph3m.LogL = -inf;
    
    hemopt2 = mopt;
    hemopt2.K = 1;
    hemopt2.N = 1;
    tmph3m = hem_h3m_c_step(tmph3m,h3m_b,hemopt2);

    if (VERBOSE_MODE >= 1)
      fprintf('vhem: K=%d, N=%d: %g\n', tmph3m.K, length(tmph3m.hmm{1}.prior), tmph3m.LogL);
    end
    
    %vhem_plot(h3m_to_hmms(tmph3m), 'ave_face120.png');
    
    % --- step 2: increase K by splitting ---
    for KK=2:mopt.K
      % find largest omega
      [~,maxo] = max(tmph3m.omega);
      
      % split it
      tmph3m.K = KK;
      
      tmp = tmph3m.omega(maxo)/2;
      tmph3m.omega(KK) = tmp;
      tmph3m.omega(maxo)  = tmp;
      
      tmph3m.hmm{KK} = tmph3m.hmm{maxo};
      [g1, g2] = split_gauss(tmph3m.hmm{maxo}.emit{1});
      tmph3m.hmm{maxo}.emit{1} = g1;
      tmph3m.hmm{KK}.emit{1} = g2;
      
      %vhem_plot(h3m_to_hmms(tmph3m), 'ave_face120.png');
      
      hemopt2.K = KK;
      tmph3m = hem_h3m_c_step(tmph3m,h3m_b,hemopt2);
      
      if (VERBOSE_MODE >= 1)
        fprintf('vhem: K=%d, N=%d: %g\n', tmph3m.K, length(tmph3m.hmm{1}.prior), tmph3m.LogL);
      end
      %vhem_plot(h3m_to_hmms(tmph3m), 'ave_face120.png');
    end
    
    % --- step 3: increase N by splitting ---
    for NN=2:mopt.N
      % for each cluster
      for j=1:mopt.K
        % find largest ROI
        if 0
          rsize = cellfun(@(x) prod(x.covars), tmph3m.hmm{j}.emit);
          [~,mi] = max(rsize);
        else
          % find most popular ROI
          [~,mi] = max(tmph3m.hmm{j}.counts_emit);
        end
        
        [g1, g2] = split_gauss(tmph3m.hmm{j}.emit{mi});
        tmph3m.hmm{j}.emit{mi} = g1;
        tmph3m.hmm{j}.emit{NN} = g2;
        
        % update prior and transition
        if 0
          tmp = tmph3m.hmm{j}.prior(mi);
          tmph3m.hmm{j}.prior(mi) = 0.55*tmp;
          tmph3m.hmm{j}.prior(NN,1) = 0.45*tmp;
          
          %tmph3m.hmm{j}.A(NN,:) = ones(1,NN-1)/(NN-1);
          tmph3m.hmm{j}.A(NN,:) = tmph3m.hmm{j}.A(mi,:);
          
          tmp = tmph3m.hmm{j}.A(:,mi);
          tmph3m.hmm{j}.A(:,mi) = 0.5*tmp;
          tmph3m.hmm{j}.A(:,NN) = 0.5*tmp;
          
          %tmph3m.hmm{j}.A(NN,NN) = 0.55*tmp(mi);
          %tmph3m.hmm{j}.A(mi,NN) = 0.45*tmp(mi);
          
          % soften
          tmph3m.hmm{j}.A = soften_A(tmph3m.hmm{j}.A, 0.01);
        else
          % uniform 
          tmph3m.hmm{j}.prior = ones(NN,1)/NN;
          tmph3m.hmm{j}.A = ones(NN,NN)/NN;
        end
      end
      
      %vhem_plot(h3m_to_hmms(tmph3m), 'ave_face120.png');
      hemopt2.N = NN;
      tmph3m = hem_h3m_c_step(tmph3m,h3m_b,hemopt2);
      
      if (VERBOSE_MODE >= 1)
        fprintf('vhem: K=%d, N=%d: %g\n', tmph3m.K, length(tmph3m.hmm{1}.prior), tmph3m.LogL);
      end
      
      %vhem_plot(h3m_to_hmms(tmph3m), 'ave_face120.png');
      %pause
    end
        
    h3m_out = tmph3m;
    
  %% initialize using random base components
  case {'base', 'baseem'}
    
    % do multiple trials and keep the best
    
    while (LL_best == -inf) || (isnan(LL_best))
    
      if (VERBOSE_MODE == 1)
        fprintf('VHEM Trial: ');
      end
      
        for t = 1 : emopt.trials

            %addpath('/home/ecoviello/cluster_gmms/gmm')
            %addpath('/home/ecoviello/cluster_gmms/gmm/hem_gmm')

            % initialize h3m
            Kb = h3m_b.K;
%            new_idx = randperm(Kb);
%            new_idx = sort(new_idx);
            
%            h3m_b1.omega = h3m_b.omega(new_idx);
%            h3m_b1.hmm = h3m_b.hmm(new_idx);
%            h3m_b1.K = h3m_b.K;
            h3m_b1 = h3m_b;
            mopt.start_time = tic;
            
            % use less data for initialization
            if h3m_b.K > 400
                inds = randi(h3m_b1.K-400);
                new_idx = inds:(inds+399);
                h3m_b2.omega = h3m_b.omega(new_idx);
                h3m_b2.omega = h3m_b2.omega / sum(h3m_b2.omega);
                h3m_b2.hmm = h3m_b.hmm(new_idx);
                h3m_b2.K = 400;
                
                h3m = initialize_hem_h3m_c(h3m_b2,mopt);
            else
                h3m = initialize_hem_h3m_c(h3m_b1,mopt);
            end
            
%             h3m = initialize_hem_h3m_c(h3m_b1,mopt);
            
            if isnumeric(h3m) && (h3m == -1)
                continue
            end

%             for smooth = [10  3  1]
% 
%                 mopt.smooth = smooth;
                h3m_new = hem_h3m_c_step(h3m,h3m_b,mopt);
%                 h3m = h3m_new;
%             end
                
            if (VERBOSE_MODE >= 2) 
              fprintf('Trial %d\t - loglikelihood: %d\n',t,h3m_new.LogL)
            elseif (VERBOSE_MODE == 1)
              fprintf('%d ', t);
            end
            
            if t == 1
                LL_best = h3m_new.LogL;
                h3m_out = h3m_new;
                t_best = t;
            elseif h3m_new.LogL > LL_best
                LL_best = h3m_new.LogL;
                h3m_out = h3m_new;
                t_best = t;
            end


        end
        
        if (re_do <= 5) &&((LL_best == -inf) || (isnan(LL_best)))
            fprintf('Need to do again... the LL was NaN ...\n')
            re_do = re_do + 1 ;
        elseif (re_do <= 10) &&((LL_best == -inf) || (isnan(LL_best)))
            fprintf('Use the ''gmm'' instead of the %s\n',mopt.initmode)
            mopt.initmode = 'gmm';
            re_do = re_do + 1 ;
        elseif (re_do > 10) &&((LL_best == -inf) || (isnan(LL_best)))
            % just give up learning
            fprintf('\n\nGIVING UP ON THIS TAG!!!!!!\n\n')
            h3m_out = h3m;
            h3m_out.given_up = 'too many trials';
            LL_out = -eps;
            t_best = 0;
            break
        end
        
    end
    
    if (VERBOSE_MODE >= 1)
      fprintf('\nBest run is %d: LL=%g\n\n',t_best,LL_best)
    end
    
end

function [g1, g2] = split_gauss(g, f)
% return 2 new gaussians, split from g
if nargin<2
  f = 1;
end

g1 = g;
g2 = g;

cv = g.covars;
switch(g.covar_type)
  case 'diag'
    [~,mi] = max(cv);
    delta_mn = zeros(size(cv));
    delta_mn(mi) = sqrt(cv(mi));
    new_cv = cv;
    new_cv(mi) = new_cv(mi) / ((2*f)^2);
end

g1.centres = g1.centres + f*delta_mn;
g2.centres = g2.centres - f*delta_mn;
g1.covars = new_cv;
g2.covars = new_cv;


function A = soften_A(A, f)

A = A + f;
A = bsxfun(@times, A, 1./sum(A,2));
