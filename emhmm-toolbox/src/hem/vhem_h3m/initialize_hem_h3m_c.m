function h3m = initialize_hem_h3m_c(h3m_b,mopt)
% 'r'       random initilization (do not use)
% 'base'    randomly select a subset of the input hmms
% 'highp'   select the input hmms with largest weight
% 'gmmNew'  
% 'gmm'     use HEM-GMM on the emission GMMs (pooled together), than
% initialize each HMM cluster center by randomizing the weight of the
% Gaussian, and with random values for the dynamics
% 'gmm_Ad'  similar to 'gmm', but set a strong diagonal component to
% transition matrices
% 'gmm_L2R' similar to 'gmm', but for left to right HMMs
%
% 'baseem' randomly select a set of base emissions
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD

h3m_K = h3m_b.K;            

if strcmp(mopt.initmode, 'baseem')
  % do nothing
else
  %%%Remove HMMs w/o correct number of states
  b = 1;
  h3m_b_t = {};
  for m = 1:h3m_K
    aaa = size(h3m_b.hmm{m}.prior,1);
    if aaa == mopt.N
      h3m_b_t.hmm{1,b} = h3m_b.hmm{1,m};
      b = b+1;
    end
  end
  b = b - 1;
  h3m_b_t.K = b;
  omegas = ones(1,h3m_b_t.K);
  h3m_b_t.omega = omegas;
  h3m_b_t.omega = h3m_b_t.omega/sum(h3m_b_t.omega);
  h3m_b = h3m_b_t;
end

mopt.Nv = 1000 * h3m_b.K;

T = mopt.tau;

if isfield(h3m_b.hmm{1}.emit{1},'nin')
    dim = h3m_b.hmm{1}.emit{1}.nin;
else
    dim = size(h3m_b.hmm{1}.emit{1}.centres,2);
end

Kb = h3m_b.K;
Kr = mopt.K;
N = mopt.N;
Nv = mopt.Nv;
M = mopt.M;

switch mopt.initmode
  
  
    % random (don't use)
case 'r'
        h3m.K = Kr;
        for j = 1 : Kr
            prior = rand(N,1);
            prior = prior/sum(prior);
            A     = rand(N);
            A     = A ./ repmat(sum(A,2),1,N);
            emit = cell(1,N);
            for n = 1 : N
                emit{n} = gmm(dim, M, mopt.emit.covar_type);
                emit{n}.covars = emit{n}.covars .* 100;
                % and modify priors
                emit{n}.priors = rand(size(emit{n}.priors));
                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
            end

            h3m.hmm{j}.prior = prior;
            h3m.hmm{j}.A     = A;
            h3m.hmm{j}.emit  = emit;
        end
        omega = rand(1,Kr);
        omega = omega/sum(omega);
        
        h3m.omega = omega;
        h3m.LogL = -inf;
  case 'baseem'
    h3m.K = Kr;
    
    for j=1:Kr
      for n=1:N
        randomb = randi(Kb);
        randomg = randi(length(h3m_b.hmm{randomb}.emit));
        %fprintf('%d,%d; ', randomb, randomg);
        h3m.hmm{j}.emit{n} = h3m_b.hmm{randomb}.emit{randomg};
      end
      h3m.hmm{j}.prior = ones(N,1)/N;
      h3m.hmm{j}.A     = ones(N,N)/N;
    end
    
    omega = ones(1,Kr)/Kr;
    h3m.omega = omega;
    h3m.LogL = -inf;
    
    
    case 'base'
        indexes = randperm(Kb);
        h3m.K = Kr;
        for j = 1 : Kr
            
            h3m.hmm{j} = h3m_b.hmm{indexes(j)};
        end
        %omega = rand(1,Kr);
        omega = ones(1,Kr);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
        
    case 'trick'
        indexes = 1:(Kb/Kr):Kb;
        h3m.K = Kr;
        for j = 1 : Kr
            
            h3m.hmm{j} = h3m_b.hmm{indexes(j)};
        end
        omega = rand(1,Kr);
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;
    
    case 'highp'
        [foo indexes] = sort(h3m_b.omega,'descend');
        h3m.K = Kr;
        for j = 1 : Kr
            
            h3m.hmm{j} = h3m_b.hmm{indexes(j)};
        end
        omega = ones(1,Kr)/Kr;
        omega = omega/sum(omega);
        h3m.omega = omega;
        h3m.LogL = -inf;

    % use hem-g3m
    
    case { 'gmmNew'}
        
        virtualSamples = Nv * Kb;
        iterations = mopt.initopt.iter;
        trials = mopt.initopt.trials;
        % fit a gaussian mixture to the data...
        
        % if there is too much input, use only some data
        
        gmms_all = cell(1,Kb*N);
        alpha = zeros(1,Kb*N);
        
        
        
        for i = 1 : h3m_b.K
            
            gmms_all( (i-1)*N+1 : (i)*N ) = h3m_b.hmm{i}.emit;
            % alpha( (i-1)*N+1 : (i)*N ) = h3m_b.hmm{i}.prior;
            p = h3m_b.hmm{i}.prior';
            A = h3m_b.hmm{i}.A;
            for t = 1 : 50
                p = p * A;
            end
            alpha( (i-1)*N+1 : (i)*N ) = p;
        end
        
        alpha = alpha/sum(alpha);
        gmms.alpha = alpha;
        gmms.mix   = gmms_all;
        
        trainMix(length(gmms_all)) = struct(gmms_all{1});
        for pippo = 1 : length(gmms_all)
            trainMix(pippo).mix = gmms_all{pippo};

        end

        [reduced_out] = GMM_MixHierEM(trainMix, Kr, virtualSamples, iterations);
        
        h3m.K = Kr;
                for j = 1 : Kr
                    prior = ones(N,1);
                    prior = prior/sum(prior);
                    
                    A = ones(N);
                    A     = A ./ repmat(sum(A,2),1,N);
                    emit = cell(1,N);
                    for n = 1 : N
                        reduced_out_use.centres = reduced_out.centres(n,:);
                        reduced_out_use.covars = reduced_out.covars(n,:);
                        reduced_out_use.ncentres = 1;
                        reduced_out_use.priors = 1;
                        reduced_out_use.nin = reduced_out.nin;
                        reduced_out_use.covar_type = reduced_out.covar_type;
                        reduced_out_use.type = reduced_out.type;
                        
                        emit{n} = reduced_out_use;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;  
                
        
    case {'g3k' 'g3n' 'gmm' 'gmm_Ad' 'gmm_L2R' 'g3k_Ad' 'g3n_Ad' 'gmm_A' 'gmm_Au'}
        
        virtualSamples = Nv * Kb;
        iterations = mopt.initopt.iter;
        trials = mopt.initopt.trials;
        % fit a gaussian mixture to the data...
        
        % if there is too much input, use only some data
        
        gmms_all = cell(1,Kb*N);
        alpha = zeros(1,Kb*N);
        
        
        
        for i = 1 : h3m_b.K
            
            gmms_all( (i-1)*N+1 : (i)*N ) = h3m_b.hmm{i}.emit;
            % alpha( (i-1)*N+1 : (i)*N ) = h3m_b.hmm{i}.prior;
            p = h3m_b.hmm{i}.prior';
            A = h3m_b.hmm{i}.A;
            for t = 1 : 50
                p = p * A;
            end
            alpha( (i-1)*N+1 : (i)*N ) = p;
        end
        
        alpha = alpha/sum(alpha);
        gmms.alpha = alpha;
        gmms.mix   = gmms_all;
        
        trainMix(length(gmms_all)) = struct(gmms_all{1});
        for pippo = 1 : length(gmms_all)
            trainMix(pippo).mix = gmms_all{pippo};

        end

        [reduced_out] = GMM_MixHierEM(trainMix, M, virtualSamples, iterations);
        
        switch mopt.initmode
            case {'gmm'}

                h3m.K = Kr;
                for j = 1 : Kr
                    prior = rand(N,1);
                    prior = prior/sum(prior);
                    A     = rand(N);
                    A     = A ./ repmat(sum(A,2),1,N);
                    emit = cell(1,N);
                    for n = 1 : N
                        emit{n} = reduced_out;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;    
                
            case {'gmm_Au'}

                h3m.K = Kr;
                for j = 1 : Kr
                    prior = ones(N,1);
                    prior = prior/sum(prior);
                    
                    A = ones(N);
                    A     = A ./ repmat(sum(A,2),1,N);
                    emit = cell(1,N);
                    for n = 1 : N
                        emit{n} = reduced_out;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;  
                
            case {'gmm_Ad'}

                h3m.K = Kr;
                for j = 1 : Kr
                    prior = rand(N,1);
                    prior = prior/sum(prior);
                    
                    % pick one A form the examples, choose one at random...
                    ind_hmm = randi(Kb);
                    A = h3m_b.hmm{ind_hmm}.A;
                    % perturb A
                    A = A + (.5/N) * A;
                    % normalize A
                    A     = A ./ repmat(sum(A,2),1,N);
                    emit = cell(1,N);
                    for n = 1 : N
                        emit{n} = reduced_out;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;  
            
            case {'gmm_A'}

                h3m.K = Kr;
                ind_hmm = randperm(Kb);
                
                for j = 1 : Kr
                    prior = h3m_b.hmm{ind_hmm(j)}.prior;
                    % pick one A form the examples, choose one at random...
                    
                    A = h3m_b.hmm{ind_hmm(j)}.A;
                    emit = cell(1,N);
                    for n = 1 : N
                        emit{n} = reduced_out;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;  
                
                
            case {'gmm_L2R'}

                h3m.K = Kr;
                for j = 1 : Kr
                    prior = rand(N,1);
                    prior = prior/sum(prior);
                    
                    % pick one A form the examples, choose one at random...
                    ind_hmm = randi(Kb);
                    A = h3m_b.hmm{ind_hmm}.A;
                    % perturb A
                    %A = A + (.5/N) * A;
                    % normalize A
                    A     = A ./ repmat(sum(A,2),1,N);
                    emit = cell(1,N);
                    for n = 1 : N
                        emit{n} = reduced_out;
                        % and randomize priors ...
                        emit{n}.priors = rand(size(emit{n}.priors));
                        emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                    end

                    h3m.hmm{j}.prior = prior;
                    h3m.hmm{j}.A     = A;
                    h3m.hmm{j}.emit  = emit;
                end
                omega = rand(1,Kr);
                omega = omega/sum(omega);
                h3m.omega = omega;
                h3m.LogL = -inf;  

                % how many gmms? as many as the number of 

            case {'g3k'} % a gmm for each hmm component ...

                        [reduced_out] = cluster_gmms_init(gmms,Kr, virtualSamples, reduced_out,iterations, trials);

                        h3m.K = Kr;
                        for j = 1 : Kr
                            prior = rand(N,1);
                            prior = prior/sum(prior);
                            A     = rand(N);
                            A     = A ./ repmat(sum(A,2),1,N);
                            emit = cell(1,N);
                            for n = 1 : N
                                emit{n} = reduced_out.mix{j};
                                % and randomize priors ...
                                emit{n}.priors = rand(size(emit{n}.priors));
                                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                            end

                            h3m.hmm{j}.prior = prior;
                            h3m.hmm{j}.A     = A;
                            h3m.hmm{j}.emit  = emit;
                        end
                        omega = rand(1,Kr);
                        omega = omega/sum(omega);
                        h3m.omega = omega;
                        h3m.LogL = -inf;
            case {'g3k_Ad'} % a gmm for each hmm component ...

                        [reduced_out] = cluster_gmms_init(gmms,Kr, virtualSamples, reduced_out,iterations, trials);

                        h3m.K = Kr;
                        for j = 1 : Kr
                            prior = rand(N,1);
                            prior = prior/sum(prior);
                            
                            % pick one A form the examples, choose one at random...
                            ind_hmm = randi(Kb);
                            A = h3m_b.hmm{ind_hmm}.A;
                            % perturb A
                            A = A + (.5/N) * A;
                            % normalize A
                            A     = A ./ repmat(sum(A,2),1,N);
                            emit = cell(1,N);
                            for n = 1 : N
                                emit{n} = reduced_out.mix{j};
                                % and randomize priors ...
                                emit{n}.priors = rand(size(emit{n}.priors));
                                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                            end

                            h3m.hmm{j}.prior = prior;
                            h3m.hmm{j}.A     = A;
                            h3m.hmm{j}.emit  = emit;
                        end
                        omega = rand(1,Kr);
                        omega = omega/sum(omega);
                        h3m.omega = omega;
                        h3m.LogL = -inf;
                        
            


            case {'g3n'} % a gmm for each state ...


                        [reduced_out] = cluster_gmms_init(gmms,N, virtualSamples,reduced_out, iterations, trials);

                        h3m.K = Kr;
                        for j = 1 : Kr
                            prior = rand(N,1);
                            prior = prior/sum(prior);
                            A     = rand(N);
                            A     = A ./ repmat(sum(A,2),1,N);
                            emit = cell(1,N);
                            for n = 1 : N
                                emit{n} = reduced_out.mix{n};
                                % and randomize priors ...
                                emit{n}.priors = rand(size(emit{n}.priors));
                                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                            end

                            h3m.hmm{j}.prior = prior;
                            h3m.hmm{j}.A     = A;
                            h3m.hmm{j}.emit  = emit;
                        end
                        omega = rand(1,Kr);
                        omega = omega/sum(omega);
                        h3m.omega = omega;
                        h3m.LogL = -inf;
            
            case {'g3n_Ad'} % a gmm for each state ...


                        [reduced_out] = cluster_gmms_init(gmms,N, virtualSamples,reduced_out, iterations, trials);

                        h3m.K = Kr;
                        for j = 1 : Kr
                            prior = rand(N,1);
                            prior = prior/sum(prior);
                            
                            % pick one A form the examples, choose one at random...
                            ind_hmm = randi(Kb);
                            A = h3m_b.hmm{ind_hmm}.A;
                            % perturb A
                            A = A + (.5/N) * A;
                            % normalize A
                            A     = A ./ repmat(sum(A,2),1,N);
                            emit = cell(1,N);
                            for n = 1 : N
                                emit{n} = reduced_out.mix{n};
                                % and randomize priors ...
                                emit{n}.priors = rand(size(emit{n}.priors));
                                emit{n}.priors = emit{n}.priors / sum(emit{n}.priors);
                            end

                            h3m.hmm{j}.prior = prior;
                            h3m.hmm{j}.A     = A;
                            h3m.hmm{j}.emit  = emit;
                        end
                        omega = rand(1,Kr);
                        omega = omega/sum(omega);
                        h3m.omega = omega;
                        h3m.LogL = -inf;
        end
            
                 
            
        
        
        
        
        
end