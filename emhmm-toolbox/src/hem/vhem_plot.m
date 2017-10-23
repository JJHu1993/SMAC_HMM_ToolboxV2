function vhem_plot(group_hmms, imgfile, mode)
% vhem_plot - plot group hmms from vhem_cluster
%
%   vhem_plot(group_hmms, imgfile, mode)
%
% INPUTS
%   group_hmms = group_hmms output from vhem_cluster
%   imgfile    = filename of image for plotting under the fixations (optional)
%   mode       = 'c' -- use compact plots [default]
%              = 'h' -- HMM plots, each row shows one group HMM
%              = 'v' -- HMM plots, each column shows one group HMM
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% VERSIONS
%  2017-01-19 - updated for duration
%             - added 'h' and 'v' options for plotting

if nargin<2
  imgfile = '';
end
if nargin<3
  mode = 'c';
end

if ~isempty(imgfile)
  img = imread(imgfile);
else
  img = [];
end

D = length(group_hmms.hmms{1}.pdf{1}.mean);
if (D==2)
  nump = 3;
else
  nump = 4;
end

% plot size
G = length(group_hmms.hmms);
switch(mode) 
  case 'c'
    % compact plot
    py = G;
    px = 1;
    plotselector = @(g,ind) g;
  case 'h'
    % plot HMM as a row
    py = G;
    px = nump;
    plotselector = @(g,ind) ind+px*(g-1);
  case 'v'
    % plot HMM as a column
    py = nump;
    px = G;
    plotselector = @(g,ind) g+px*(ind-1);
end

figure
for g=1:G
  %title(['prior ' glab], 'interpreter', 'none');
  
  switch(mode)
    case 'c'
      ind = 1;
      subplot(py,px,plotselector(g,ind))
      vbhmm_plot_compact(group_hmms.hmms{g}, imgfile);
      title(sprintf('Group %d (size=%d)', g, group_hmms.group_size(g)));
  
    case {'v', 'h'}
      ind = 1;
      subplot(py,px,plotselector(g,ind))
      plot_emissions([], [], group_hmms.hmms{g}.pdf, img)
      title(sprintf('Group %d (size=%d)', g, group_hmms.group_size(g)));
      ind = ind+1;
      
      if (D>2)
        subplot(py,px,plotselector(g,ind))
        plot_emissions_dur([], [], group_hmms.hmms{g}.pdf);
        ind = ind+1;
      end
      
      subplot(py,px,plotselector(g,ind))
      plot_transprob(group_hmms.hmms{g}.trans)
      ind = ind+1;
    
      subplot(py,px,plotselector(g,ind))
      plot_prior(group_hmms.hmms{g}.prior)
      ind = ind+1;
  
    otherwise
      error('bad option');
  end
end
