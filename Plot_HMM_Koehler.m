%If you use this toolbox, please cite Coutrot et al., 
%"Scanpath modeling and classification with Hidden Markov Models", Behavior
%Research Methods, 2017
% Compute 1 HMM per scanpath, or 1 HMM for a group of scanpaths
clear
close all
addpath ./stimuli/Images_Koehler/ % where the stimuli are stored
addpath(genpath('emhmm-toolbox'))
load EyeData_Koehler % contains eye positions from Koehler et al., JoV 2014

% variational approach: automatically selects optimal state number from K = 1 to 3
K = 1:3;

%plot the HMM states on the stimuli
isplotHMM=1;

%% Read stimuli
istim=50;
im_name=['image_r_' num2str(istim) '.jpg'];
im_name_struct=['image_r_' num2str(istim)];
im= imread(im_name);

%% Learn 1 HMM per scanpath
for isub=1:9
    % Extract scanpath of observer isub
    scanpath = extract_scanpath(example_EyeData,'objsearch',isub,istim,K);
    if ~isempty(scanpath{1,1})
        % Compute corresponding HMM
        vbopt=initialize_HMM_computation(im);
        [hmm,~] = vbhmm_learn(scanpath, K, vbopt);
        
        % sort states from left to right
        hmm = sort_hmm_state(hmm);
        
        if isplotHMM
            subplot(3,3,isub)
            plot_hmm_state(hmm,scanpath,im)
            s=sprintf('observer %u',isub);
            title(s)
            pause(0.01)
        end
    end
end


%% Learn 1 HMM from a group of scanpaths
%Select observers 
observers=1:19;
% Extract their scanpath
scanpath = extract_scanpath(example_EyeData,'objsearch',observers,istim,K);
%remove empty cell array contents
scanpath=scanpath(~cellfun('isempty',scanpath));

if ~isempty(scanpath{1,1})
    % Compute corresponding HMM
    vbopt=initialize_HMM_computation(im);
    [hmm,~] = vbhmm_learn(scanpath, K, vbopt);
    
    % sort states from left to right
    hmm = sort_hmm_state(hmm);
    
    if isplotHMM
        vbhmm_plot(hmm,scanpath,im_name)
        pause(0.01)
    end
end