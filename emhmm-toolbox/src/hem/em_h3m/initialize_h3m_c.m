function h3m = initialize_h3m_c(observations,mopt,K,N,M)
% 'r'    random initialization of all parameters (this will probably never work well for you)
% 'p'    randomly partition the input time series in K group, and estimate each HMM component on one of the partition
% 'g'    first estimate a GMM on all the data, than initialize each HMM by setting emission to the GMM (with randomized weights) and using random parameters for the HMM dynamics
% 'gm'   similar to 'g', but uses MATLAB own function 
% 'km'   similar to 'g' but uses k means
% 'gL2R' similar to 'g'. but initialize HMMs as left to righ HMMs
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

[T,dim] = size(observations{1});

K = mopt.K;
N = mopt.N;
M = mopt.M;

switch mopt.initmode
    case 'r'  % RANDOM
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = gmm(dim, M, mopt.emit.covar_type);
                % and modify priors
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;

    case 'p' % estimate each HMM component on a random partition
        h3m.K = K;
        I = length(observations);
        indexes = randperm(I);
        n_o = floor(I/K);
        for j = 1 : K
            
            
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = gmm(dim, M, 'diag');
                % and modify priors
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            
            h3m_foo.hmm{1}.prior = prior;
            h3m_foo.hmm{1}.A     = A;
            h3m_foo.hmm{1}.emit     = emit;
            h3m_foo.K            = 1;
            h3m_foo.omega        = 1;
            h3m_foo.LogL         = -inf;
            h3m_foo.max_iter     = 100;
            h3m_foo.termvalue    = 10e-4;
            
            mopt_foo = mopt;
            h3m_new = em_h3m_c_step(h3m_foo,observations( indexes((j-1)*n_o+1 : j*n_o )),mopt_foo );
            
%             h3m_new = h3m_regularize(h3m_new,0.00001);
            
%             % avoid problems for later
%             BB = h3m_new.hmm{1}.B;
%             if any( (BB(:) == 0) )
%                 min_b = min(BB(:) + (BB(:) == 0));
%                 BB(find(BB == 0)) = min_b * 0.0001;
%                 BB     = BB ./ repmat(sum(BB,2),1,M);
%                 h3m_new.hmm{1}.B = BB;
%             end
            
            h3m.hmm{j} = h3m_new.hmm{1};
        end
        omega = (1/K)*ones(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
    
    case 'g' % ESTIMATE GMM on the DATA, than for each HMM randomize GMM weights and initialize random HMM dynamics
        % fit a gaussian mixture to the data...
        [T dim] = size(observations{1});
        L = min(length(observations),200);
        inds  = randperm(length(observations));
        observations = observations(inds(1:L));
        
        features = zeros([length(observations) 1].*[2*T dim]);
        
        ind_used = 0;
        for p = 1 : length(observations)
            [T] = size(observations{p},1);
            features(ind_used+1:ind_used+T,:) = observations{p};
            ind_used = ind_used + T;
         end 
        
        features(ind_used+1:end,:) =[];
        
        if M > 1
            [mix Li] = GMM_EM3(features, M);
        else
            mix.priors  = 1;
            mix.centres = mean(features);
            mix.covars  = var(features) + 0.01;
            mix.covar_type = 'diag';
            mix.ncentres = 1;
            mix.type = 'gmm';
            mix.nin = size(features,2);
        end
        
        
        
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = mix;
                % and randomize priors ...
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
        
case 'gm' % similar to 'g'. but uses MATLAB own function 
        % fit a gaussian mixture to the data...
        [T dim] = size(observations{1});
        L = min(length(observations),200);
        inds  = randperm(length(observations));
        observations = observations(inds(1:L));
        
        features = zeros([length(observations) 1].*[2*T dim]);
        
        ind_used = 0;
        for p = 1 : length(observations)
            [T] = size(observations{p},1);
            features(ind_used+1:ind_used+T,:) = observations{p};
            ind_used = ind_used + T;
         end 
        
        features(ind_used+1:end,:) =[];
        
        
        Li_best = -inf;
        
        for tii = 1 : 10
            [mixti Li] = GMM_EM3(features, M*N);
            if Li> Li_best
                Li_best = Li;
                mix = mixti;
            end
            
        end
        
        features = features + random('norm',0,.0001,size(features));
        obj = gmdistribution.fit(features,M*N,'CovType','diagonal','SharedCov',0==1,'Regularize',0.001);
        centres = squeeze(obj.mu);
        covars  = squeeze(obj.Sigma);
        priors  = obj.PComponents;
        
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = mix;
                emit{n}.ncentres = M;
                emit{n}.centres = centres((n-1)+1:(n-1)+M,:); %mix.centres((n-1)+1:(n-1)+M,:);
                emit{n}.covars = covars((n-1)+1:(n-1)+M,:);   %mix.covars((n-1)+1:(n-1)+M,:);
                % and randomize priors ...
                emit{n}.priors = priors((n-1)+1:(n-1)+M); %mix.priors((n-1)+1:(n-1)+M);
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
        
case 'km' % as 'g' but uses k means
        
        [T dim] = size(observations{1});
        L = min(length(observations),200);
        inds  = randperm(length(observations));
        observations = observations(inds(1:L));
        
        features = zeros([length(observations) 1].*[2*T dim]);
        
        
        ind_used = 0;
        for p = 1 : length(observations)
            [T] = size(observations{p},1);
            features(ind_used+1:ind_used+T,:) = observations{p};
            ind_used = ind_used + T;
         end 
        
        features(ind_used+1:end,:) =[];
        
        
        Li_best = -inf;
        
        [centres,mincenter] = arKmeans(features, M*N,1,0);
        [mix Li] = GMM_EM3(features, M*N);
        covars  = var(features);
%         for tii = 1 : 10
%             [mixti Li] = GMM_EM3(features, M*N);
%             if Li> Li_best
%                 Li_best = Li;
%                 mix = mixti;
%             end
%             
%         end
        
        
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = mix;
                emit{n}.ncentres = M;
                emit{n}.centres = centres((n-1)+1:(n-1)+M,:);
                emit{n}.covars = covars;
                % and randomize priors ...
                emit{n}.priors = mix.priors((n-1)+1:(n-1)+M);
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
        
    case 'gnew'
        % fit a gaussian mixture to the data...
        [T dim] = size(observations{1});
        L = min(length(observations),200);
        inds  = randperm(length(observations));
        observations = observations(inds(1:L));
        
        features = zeros([length(observations) 1].*[2*T dim]);
        
        ind_used = 0;
        for p = 1 : length(observations)
            [T] = size(observations{p},1);
            features(ind_used+1:ind_used+T,:) = observations{p};
            ind_used = ind_used + T;
         end 
        
        features(ind_used+1:end,:) =[];
        
        [mix Li] = GMM_EM3(features, N);
        mix.covars =  mix.covars + 0.001;
        
%         if M > 1
%             [mix Li] = GMM_EM3(features, M);
%         else
%             mix.priors  = 1;
%             mix.centres = mean(features);
%             mix.covars  = var(features) + 0.01;
%             mix.covar_type = 'diag';
%             mix.ncentres = 1;
%             mix.type = 'gmm';
%             mix.nin = size(features,2);
%         end
        
        
        
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                mix_a = mix;
                mix_a.covars = mix_a.covars(n,:);
                mix_a.centres = mix_a.centres(n,:);
                mix_a.priors = 1;
                mix_a.numcentres = 1;
                emit{n} = mix;
                % and randomize priors ...
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
    case 'gL2R' % left to righ models
        % fit a gaussian mixture to the data...
        [T dim] = size(observations{1});
        L = min(length(observations),200);
        inds  = randperm(length(observations));
        observations = observations(inds(1:L));
        features = zeros([length(observations) 1].*[T dim]);
        for p = 1 : length(observations)
            features((p-1)*T+1:p*T,:) = observations{p};
        end 
        
        if M > 1
            [mix Li] = GMM_EM3(features, M);
        else
            mix.priors  = 1;
            mix.centres = mean(features);
            mix.covars  = var(features);
            mix.covar_type = 'diag';
            mix.ncentres = 1;
            mix.type = 'gmm';
            mix.nin = size(features,2);
        end
        mix.nin = size(features,2);
        
        
        h3m.K = K;
        for j = 1 : K
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A = triu(A);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = mix;
                % and randomize priors ...
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,K);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
end