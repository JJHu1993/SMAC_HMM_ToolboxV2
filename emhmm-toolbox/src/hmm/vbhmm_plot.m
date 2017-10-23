function vbhmm_plot(hmm, data, imgfile, grpnames)
% vbhmm_plot - plot VBHMM with data
%
%   vbhmm_plot(hmm, data, imgfile, grpnames)
%
% INPUTS
%   hmm      = hmm from vbhmm_learn
%   data     = data, same format as used for vbhmm_learn
%   imgfile  = filename of image for plotting under the fixations (optional)
%   grpnames = names of the groups if using groups for vbhmm_learn (optional)
%              cell array, where grpnames{g} = name of g-th group.
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2016-11-21: ABC - moved helper plotting code into separate files (for re-usability)
% 2017-01-17: ABC - update for fixation duration; change fixation marker to circles

if nargin<3
  imgfile = '';
end
if nargin<4
  grpnames = {};
end

if ~isempty(imgfile)
  img = imread(imgfile);
else
  img = [];
end

D = length(hmm.pdf{1}.mean);

if isfield(hmm, 'group_ids')
  usegroups = 1;
  py = 1;
else
  usegroups = 0;
  py = 2;
end

if usegroups
  % if using groups, show comparison plots for each parameter
  G = length(hmm.group_ids);
  px=2;
  py=ceil(G/px);
  
  fig1 = figure;
  fig2 = figure;
  fig3 = figure;
  fig4 = figure;
  fig5 = figure;
  
  if D>2
    fig6 = figure;
  end
  
  for g=1:G
    if ~isempty(grpnames)
      glab = grpnames{g};
    else
      glab = sprintf('group %d', hmm.group_ids(g));
    end
    gdata = data(hmm.group_inds{g});
    ggamma = hmm.gamma(hmm.group_inds{g});
    gN     = hmm.Ng{g}; %sum(cat(2,ggamma{:}), 2);    
    
    figure(fig1)
    subplot(py,px,g);
    plot_fixations(gdata, img, 0);
    title(['fixations ' glab], 'interpreter', 'none');
  
    figure(fig2)
    subplot(py,px,g);
    plot_emissions(gdata, ggamma, hmm.pdf, img)
    title(['emissions ', glab], 'interpreter', 'none');
    
    figure(fig3)
    subplot(py,px,g);
    plot_emcounts(gN);
    title(['counts ' glab], 'interpreter', 'none');
    
    figure(fig4)
    subplot(py,px,g);
    plot_transprob(hmm.trans{g})
    title(['trans ' glab], 'interpreter', 'none');
        
    figure(fig5)
    subplot(py,px,g);
    plot_prior(hmm.prior{g})
    title(['prior ' glab], 'interpreter', 'none');
    
    if (D>2)
      figure(fig6)
      subplot(py,px,g)
      plot_emissions_dur(data, ggamma, hmm.pdf)
      title(['emissions duration ', glab], 'interpreter', 'none');
    end

  end
  
  return
end

if D==2
  px = 3;
else
  px = 4;
end


%% plot a single HMM
figure
ind = 1;
subplot(py,px,ind)
plot_fixations(data, img, hmm.LL);
ind = ind+1;

subplot(py,px,ind)
plot_emissions(data, hmm.gamma, hmm.pdf, img)
ind = ind+1;

if (D>2)
  % plot duration emissions
  subplot(py,px,ind)
  plot_emissions_dur(data, hmm.gamma, hmm.pdf);
  ind = ind+1;
end

subplot(py,px,ind)
plot_emcounts(hmm.N);
ind = ind+1;

%if usegroups
%  % now show each group as a separate HMM w/ group data
%  for g=1:length(hmm.group_ids)
%    % extract the group-specific transition & prior
%    ghmm = hmm;
%    ghmm.N1 = hmm.N1{g};
%    ghmm.M  = hmm.M{g};
%    ghmm.prior = hmm.prior{g};
%    ghmm.trans = hmm.trans{g};
%    ghmm.gamma = {hmm.gamma{hmm.group_inds{g}}};
%    ghmm.N     = sum(cat(2,ghmm.gamma{:}), 2);
%    ghmm = rmfield(ghmm, {'group_ids', 'group_inds', 'group_map'});
%    gdata = {data{hmm.group_inds{g}}};
%    
%    vbhmm_plot(ghmm,gdata,imgfile);
%    % change title
%    subplot(2,3,1)
%    title(sprintf('group %d', hmm.group_ids(g)));
%  end
%  return
%end

subplot(py,px,ind)
plot_transcount(hmm.M)
ind = ind+1;

subplot(py,px,ind)
plot_transprob(hmm.trans)
ind = ind+1;

subplot(py,px,ind)
plot_prior(hmm.prior)
ind = ind+1;

