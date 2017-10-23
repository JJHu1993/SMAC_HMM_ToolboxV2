function vhem_plot_clusters(group_hmms, hmms, imgfile, plotmode, sortmode)
% vhem_plot_clusters - plot cluster membership and group HMMs
%
%   vhem_plot_clusters(group_hmms, hmms, imgfile, plotmode, sortmode)  
%
%   INPUT:
%     group_hmms = group HMMs, i.e., the output of vhem_cluster
%     hmms       = cell array of individual HMMs
%     imgfile    = image file for visualization
%     plotmode   = compact plotting mode (see vbhmm_plot_compact)
%     sortmode   = ROI sorting (see vbhmm_standardize)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2016-11-22: ABC - created

if nargin<4
  plotmode = 'r';
end
if nargin<5
  sortmode = 'f';
end

for g=1:length(group_hmms.hmms)
  N = 1 + group_hmms.group_size(g);
  py = ceil(sqrt(N));
  px = ceil(N / py);
  
  figure
  subplot(py,px,1)
  vbhmm_plot_compact(vbhmm_standardize(group_hmms.hmms{g}, sortmode), imgfile, plotmode);
  title(sprintf('Group %d', g));
  
  for i=1:N-1
    subplot(py,px,i+1);
    x = group_hmms.groups{g}(i);
    vbhmm_plot_compact(vbhmm_standardize(hmms{x}, sortmode), imgfile, plotmode);
    title(sprintf('Subj %d', x));
  end
  
end