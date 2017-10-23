% demo_faces_jov_duration - example of HMMs with fixation location and duration
%
% ---
% For each subject, we train an HMM. The subjects' HMMs are clustered
% using VHEM to obtain the common strategies used.
%
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-19
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

clear
close all

% set random state to be able to replicate results
rand('state', 101);
randn('state', 101);

%% Load data with fixation location and duration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data,SubjectIDs,TrialIDs] = read_xls_fixations('jov_duration.xls');

% data is stored in a cell array
% data{i}         = i-th subject
% data{i}{j}      = ... j-th trial
% data{i}{j}(t,:) = ... [x y d] location of t-th fixation. duration "d" is in milliseconds 

% data is also in this mat file:
%load jov_duration.mat 


% the number of subjects
N = length(data);
N = 10;  % only look at 10 in this demo

%% VB Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = 2:3;
vbopt.alpha = 1;
vbopt.mu    = [160;210;250];  % the image center, and 250ms duration
vbopt.W     = [0.001, 0.001, 0.0016]; % stdev of 31 pixels for ROIs, and stddev of 25ms for duration
vbopt.beta  = 1;
vbopt.v     = 10; 
vbopt.epsilon = 1;
vbopt.showplot = 0;

faceimg = 'ave_face120.png';


vbopt = vbhmm_auto_hyperparam(vbopt, {data{1:N}}, faceimg, 'd')


%% Learn Subject's HMMs %%%%%%%%%%%%%%%%%%%%%
% estimate for each subject
for i=1:N
  fprintf('=== running Subject %d ===\n', i);
  hmms{i} = vbhmm_learn(data{i}, K, vbopt);
    
  vbhmm_plot(hmms{i}, data{i}, faceimg);
  drawnow  
end

% plot each subject
for i=1:N
  if mod(i,16)==1
    figure
  end
  subplot(4,4,mod(i-1,16)+1)
  vbhmm_plot_compact(hmms{i}, faceimg);
  title(sprintf('Subject %d', i));
end


%% Run HEM clustering (1 cluster) %%%%%%%%%%%%%%%%
fprintf('=== Clustering 1 ===\n');
hemopt.sortclusters = 'f';
[group_hmms1] = vhem_cluster(hmms, 1, 3, hemopt)

% plot the groups
vhem_plot(group_hmms1, faceimg);

%% Run HEM Clustering (2 clusters) %%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('=== Clustering 2 ===\n');
[group_hmms2] = vhem_cluster(hmms, 2, 3, hemopt)

% plot the groups
vhem_plot(group_hmms2, faceimg);

% show group membership
fprintf('Group membership: \n');
for j=1:length(group_hmms2.groups)
  fprintf('  group %d = %s\n', j, mat2str(group_hmms2.groups{j}));
end


