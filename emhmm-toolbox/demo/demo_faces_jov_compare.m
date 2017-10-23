% demo_faces_jov_compare - example of comparing HMMs
%
% ---
% For each subject, we separated trials that led to correct responses
% and trials that led to incorrect (wrong) responses.
% A separate HMM is learned for correct trials and wrong trials.
% Then, we compare the "correct" HMM and "wrong" HMM.
%
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


clear
close all

% set random state to be able to replicate results
rand('state', 101);
randn('state', 101);

%% Load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load jov_data.mat 
% jov_data contains the data that was used for 
% Chuk T., Chan, A. B., & Hsiao, J. H. (2014). 
% Understanding eye movements in face recognition using
% hidden Markov models. Journal of Vision 14(8).
% doi:10.1167/14.11.8.

% data is stored in a cell array
% data{i}         = i-th subject
% data{i}{j}      = ... j-th trial
% data{i}{j}(t,:) = ... [x y] location of t-th fixation 

% the number of subjects
N = length(data);


%% VB Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = 2:3; 
vbopt.alpha = 1;
vbopt.mu    = [160;210]; 
vbopt.W     = 0.001; 
vbopt.beta  = 1; 
vbopt.v     = 10; 
vbopt.epsilon = 1;
vbopt.showplot = 0;

faceimg = 'ave_face120.png';

%% Learn Subject's HMMs %%%%%%%%%%%%%%%%%%%%%
% estimate for each subject
for i=1:N
  fprintf('=== running Subject %d ===\n', i);
  
  % learn HMM for correct trials
  hmms_correct{i} = vbhmm_learn(data_correct{i}, K, vbopt);
  
  % learn HMM for wrong trials
  hmms_wrong{i}   = vbhmm_learn(data_wrong{i}, K, vbopt);
  
  figure(100)
  clf
  subplot(1,2,1)
  vbhmm_plot_compact(hmms_correct{i}, faceimg);
  title(sprintf('Subject %d - correct', i));
  subplot(1,2,2)
  vbhmm_plot_compact(hmms_wrong{i}, faceimg);
  title(sprintf('Subject %d - wrong', i));
  
  drawnow
end

%% Run statistical tests %%
% see if correct HMMs are different from wrong HMMs.
fprintf('=== correct vs. wrong ===\n');
[p, info, lld] = stats_ttest(hmms_correct, hmms_wrong, data_correct);
p
info

fprintf('=== wrong vs. correct ===\n');
[p, info, lld] = stats_ttest(hmms_wrong, hmms_correct, data_wrong);
p 
info
return


