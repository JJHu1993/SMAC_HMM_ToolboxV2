% demo_faces - example of eye gaze analysis for face recognition 
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

clear
close all

% set random state to be able to replicate results
rand('state', 101);
randn('state', 101);


%% Load data from xls %%%%%%%%%%%%%%%%%%%%%%%%%%
% see the xls file for the format
[data, SubjNames, TrialNames] = read_xls_fixations('demodata.xls');

% the data is read and separated by subject and trial, and stored in a cell array:
% data{i}         = i-th subject
% data{i}{j}      = ... j-th trial
% data{i}{j}(t,:) = ... [x y] location of t-th fixation 

% the same data is stored in a mat file.
% load demodata.mat 

% the number of subjects
N = length(data);


%% VB Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

K = 2:3; % automatically select from K=2 to 3
vbopt.alpha = 0.1;
vbopt.mu    = [256;192];
vbopt.W     = 0.005;
vbopt.beta  = 1;
vbopt.v     = 5;
vbopt.epsilon = 0.1;
vbopt.showplot = 0;
faceimg = 'face.jpg';


%% Learn Subject's HMMs %%%%%%%%%%%%%%%%%%%%%
% estimate for each subject
for i=1:N
  fprintf('=== running Subject %d ===\n', i);
  hmms{i} = vbhmm_learn(data{i}, K, vbopt);
end

% show subject 1
vbhmm_plot(hmms{1}, data{1}, faceimg);
figure, vbhmm_plot_compact(hmms{1}, faceimg);


% plot each subject
figure
for i=1:N
  subplot(4,3,i)
  vbhmm_plot_compact(hmms{i}, faceimg);
  title(sprintf('Subject %d', i));
end


%% Run HEM clustering (1 group) %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% summarize all subjects with one HMM
fprintf('=== Clustering (1 group) ===\n');
hemopt.sortclusters = 'f';
[all_hmms1] = vhem_cluster(hmms, 1, 3, hemopt)  % 1 group, 3 hidden states

% plot the overall HMM
vhem_plot(all_hmms1, faceimg);


%% Run HEM Clustering (2 groups) %%%%%%%%%%%%%%%%%%%%%%%%%%
% cluster subjects into 2 groups
fprintf('=== Clustering (2 groups) ===\n');
[group_hmms2] = vhem_cluster(hmms, 2, 3, hemopt)  % 2 groups, 3 hidden states

% plot the groups
vhem_plot(group_hmms2, faceimg);

% plot the groups and cluster members
vhem_plot_clusters(group_hmms2, hmms, faceimg);

% show group membership
fprintf('Group membership: \n');
for j=1:length(group_hmms2.groups)
  fprintf('  group %d = %s\n', j, mat2str(group_hmms2.groups{j}));
end


%% Statistical test %%%%%%%%%%%%%%%%%%%%%%%%%%%
% collect data for group 1 and group 2
data1 = {data{group_hmms2.groups{1}}};
data2 = {data{group_hmms2.groups{2}}};

% run t-test for hmm1 
[p, info, lld] = stats_ttest(group_hmms2.hmms{1}, group_hmms2.hmms{2}, data1);
fprintf('test group hmm1 different from group hmm2: t(%d)=%g; p=%g\n', info.df, info.tstat, p);

% run t-test for hmm2
[p, info, lld] = stats_ttest(group_hmms2.hmms{2}, group_hmms2.hmms{1}, data2);
fprintf('test group hmm2 different from group hmm1: t(%d)=%g; p=%g\n', info.df, info.tstat, p);



