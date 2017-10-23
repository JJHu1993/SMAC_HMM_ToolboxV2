%If you use this toolbox, please cite Coutrot et al., 
%"Scanpath modeling and classification with Hidden Markov Models", Behavior
%Research Methods, 2017

% Compute HMM-based gaze descriptors for Koehler's dataset (static images, 3 tasks)
clear
close all
addpath ./stimuli/Frames_Coutrot/ % where the stimuli are stored
addpath(genpath('emhmm-toolbox'))
load EyeData_Coutrot % contains eye positions from Coutrot & Guyader, JoV 2014

% variational approach: automatically selects optimal state number from K = 1 to 3
K = 1:3;
% [priors transition_matrix_coeff state_centre state_cov]
param_nb=max(K)+max(K)^2+max(K)*2+max(K)*2;
%plot the HMM states on the stimuli
isplotHMM=1;

nstim=size(example_EyeData_Coutrot.with_os.x_pos,2); %number of stimuli
for istim=1:nstim
    fprintf('stim %u\n',istim)
    %close all
    
    % Read stimuli
    im_name=['clip_' num2str(istim) '.jpg'];
    im_name_struct=['clip__' num2str(istim)];
    im= imread(im_name);
    
    % Maximum number of observer per task
    maxsub=max([size(example_EyeData_Coutrot.with_os.x_pos,1),size(example_EyeData_Coutrot.without_os.x_pos,1)]);
    % Initialize gaze descriptor vector
    gaze_descriptor_ws=NaN(maxsub,param_nb);
    gaze_descriptor_wos=NaN(maxsub,param_nb);
    
    % Loop on subjects
    for isub=1:maxsub
        
        %% Auditory Condition 1: With Orginal Soundtrack
        
        % Extract current scanpath
        scanpath_ws = extract_scanpath(example_EyeData_Coutrot,'with_os',isub,istim,K);
        if ~isempty(scanpath_ws{1,1})
            % Compute corresponding HMM
            vbopt=initialize_HMM_computation(im);
            [hmm_ws,~] = vbhmm_learn(scanpath_ws, K, vbopt);
            
            % sort states from left to right
            hmm_ws = sort_hmm_state(hmm_ws);
            
            % add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension
            hmm_ws=pad_with_ghost_states(hmm_ws,max(K),im);
            
            %Extract gaze_descriptor vector from HMM parameters: priors, transition matrix coefficients, state centres and state covariances
            gaze_descriptor_ws(isub,:) =extract_hmm_parameters(hmm_ws);
            
            if isplotHMM
                subplot(1,3,1)
                plot_hmm_state(hmm_ws,scanpath_ws,im)
                title WithSound
            end
        end
        %% Auditory Condition 2: Without Orginal Soundtrack
        
        % Extract current scanpath
        scanpath_wos = extract_scanpath(example_EyeData_Coutrot,'without_os',isub,istim,K);
        if ~isempty(scanpath_wos{1,1})
            % Compute corresponding HMM
            vbopt=initialize_HMM_computation(im);
            vbopt.do_constrain_var=1;
            [hmm_wos,~] = vbhmm_learn(scanpath_wos, K, vbopt);
            
            % sort states from left to right
            hmm_wos = sort_hmm_state(hmm_wos);
            
            % add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension
            hmm_wos=pad_with_ghost_states(hmm_wos,max(K),im);
            
            %Extract gaze_descriptor vector from HMM parameters:
            %priors, transition matrix coefficients, state centres and state covariances
            gaze_descriptor_wos(isub,:) =extract_hmm_parameters(hmm_wos);
            
            if isplotHMM
                subplot(1,3,2)
                plot_hmm_state(hmm_wos,scanpath_wos,im)
                title WithoutSound
                pause(0.1)
            end
        end
    end
    
    gaze_descriptor_ws(isnan(gaze_descriptor_ws(:,1)),:)=[];
    gaze_descriptor_wos(isnan(gaze_descriptor_wos(:,1)),:)=[];
    
    HMM_descriptor_Coutrot.(im_name_struct).with_os.gaze_descriptor=gaze_descriptor_ws;
    HMM_descriptor_Coutrot.(im_name_struct).without_os.gaze_descriptor=gaze_descriptor_wos;
    
end
save('HMM_descriptor_Coutrot','HMM_descriptor_Coutrot')


