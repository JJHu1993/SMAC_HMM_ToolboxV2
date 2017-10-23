%If you use this toolbox, please cite Coutrot et al., 
%"Scanpath modeling and classification with Hidden Markov Models", Behavior
%Research Methods, 2017

% Compute HMM-based gaze descriptors for Koehler's dataset (static images, 3 tasks)
clear
close all
addpath ./stimuli/Images_Koehler/ % where the stimuli are stored
addpath(genpath('emhmm-toolbox'))
load EyeData_Koehler % contains eye positions

% variational approach: automatically selects optimal state number from K = 1 to 3
K = 1:3;
% [priors transition_matrix_coeff state_centre state_cov]
param_nb=max(K)+max(K)^2+max(K)*2+max(K)*2; 
%plot the HMM states on the stimuli
isplotHMM=1;

%nstim=size(example_EyeData.freeview.x_pos,2); %number of stimuli
nstim=50;%here we just take the first 50.
for istim=1:nstim
    fprintf('stim %u\n',istim)
    %close all
    
    % Read stimuli
    im_name=['image_r_' num2str(istim) '.jpg'];
    im_name_struct=['image_r_' num2str(istim)];
    im= imread(im_name);
    
    % Maximum number of observer per task
    maxsub=max([size(example_EyeData.freeview.x_pos,1),size(example_EyeData.salview.x_pos,1),size(example_EyeData.objsearch.x_pos,1)]);
    % Initialize gaze descriptor vector
    gaze_descriptor_free=NaN(maxsub,param_nb);
    gaze_descriptor_sal=NaN(maxsub,param_nb);
    gaze_descriptor_obj=NaN(maxsub,param_nb);
    
    % Loop on subjects
    for isub=1:maxsub
        
        %% Task 1: Free Viewing
        
        % Extract current scanpath
        scanpath_free = extract_scanpath(example_EyeData,'freeview',isub,istim,K);
        if ~isempty(scanpath_free{1,1})
            % Compute corresponding HMM
            vbopt=initialize_HMM_computation(im);
            [hmm_free,~] = vbhmm_learn(scanpath_free, K, vbopt);
            
            % sort states from left to right
            hmm_free = sort_hmm_state(hmm_free);
            
            % add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension
            hmm_free=pad_with_ghost_states(hmm_free,max(K),im);
            
            %Extract gaze_descriptor vector from HMM parameters: priors, transition matrix coefficients, state centres and state covariances
            gaze_descriptor_free(isub,:) =extract_hmm_parameters(hmm_free);
            
            if isplotHMM
                subplot(1,3,1)
                plot_hmm_state(hmm_free,scanpath_free,im)
                title FreeViewing
            end
        end
        
        %% Task 2: Saliency Viewing
        
        % Extract current scanpath
        scanpath_sal = extract_scanpath(example_EyeData,'salview',isub,istim,K);
        if ~isempty(scanpath_sal{1,1})
            % Compute corresponding HMM
            vbopt=initialize_HMM_computation(im);
            [hmm_sal,~] = vbhmm_learn(scanpath_sal, K, vbopt);
            
            % sort states from left to right
            hmm_sal = sort_hmm_state(hmm_sal);
            
            % add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension
            hmm_sal=pad_with_ghost_states(hmm_sal,max(K),im);
            
            %Extract gaze_descriptor vector from HMM parameters: priors, transition matrix coefficients, state centres and state covariances
            gaze_descriptor_sal(isub,:) =extract_hmm_parameters(hmm_sal);
            
            if isplotHMM
                subplot(1,3,2)
                plot_hmm_state(hmm_sal,scanpath_sal,im)
                title SalViewing
            end
        end
        
        %% Task 3: Object Search
        
        % Extract current scanpath
        scanpath_obj = extract_scanpath(example_EyeData,'objsearch',isub,istim,K);
        if ~isempty(scanpath_obj{1,1})
            % Compute corresponding HMM
            vbopt=initialize_HMM_computation(im);
            [hmm_obj,~] = vbhmm_learn(scanpath_obj, K, vbopt);
            
            % sort states from left to right
            hmm_obj = sort_hmm_state(hmm_obj);
            
            % add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension
            hmm_obj=pad_with_ghost_states(hmm_obj,max(K),im);
            
            %Extract gaze_descriptor vector from HMM parameters: priors, transition matrix coefficients, state centres and state covariances
            gaze_descriptor_obj(isub,:) =extract_hmm_parameters(hmm_obj);
            
            if isplotHMM
                subplot(1,3,3)
                plot_hmm_state(hmm_obj,scanpath_obj,im)
                title ObjSearch
                pause(0.01)
            end
        end
        
    end
    %Remove NaNs
    gaze_descriptor_free(isnan(gaze_descriptor_free(:,1)),:)=[];
    gaze_descriptor_sal(isnan(gaze_descriptor_sal(:,1)),:)=[];
    gaze_descriptor_obj(isnan(gaze_descriptor_obj(:,1)),:)=[];
    %Create structure
    HMM_descriptor_Koehler.(im_name_struct).freeview.gaze_descriptor=gaze_descriptor_free;
    HMM_descriptor_Koehler.(im_name_struct).salview.gaze_descriptor=gaze_descriptor_sal;
    HMM_descriptor_Koehler.(im_name_struct).objsearch.gaze_descriptor=gaze_descriptor_obj;
   
end
save('HMM_descriptor_Koehler','HMM_descriptor_Koehler')