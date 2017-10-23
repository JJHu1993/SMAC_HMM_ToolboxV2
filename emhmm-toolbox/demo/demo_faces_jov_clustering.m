% demo_faces_jov_clustering - example of HMM clustering
%
% ---
% For each subject, we train an HMM. The subjects' HMMs are clustered
% using VHEM to obtain the common strategies used.
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
% jov_data contains the data that was used in 
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
  hmms{i} = vbhmm_learn(data{i}, K, vbopt);
  
  figure(100)
  clf
  vbhmm_plot_compact(hmms{i}, faceimg);
  title(sprintf('Subject %d', i));
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


%% Run HEM Clustering (3 clusters) %%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('=== Clustering 3 ===\n');
[group_hmms3] = vhem_cluster(hmms, 3, 3, hemopt)

% plot the groups
vhem_plot(group_hmms3, faceimg);

% plot the groups and cluster members
%vhem_plot_clusters(group_hmms, hmms, faceimg);

% show group membership
fprintf('Group membership: \n');
for j=1:length(group_hmms3.groups)
  fprintf('  group %d = %s\n', j, mat2str(group_hmms3.groups{j}));
end
