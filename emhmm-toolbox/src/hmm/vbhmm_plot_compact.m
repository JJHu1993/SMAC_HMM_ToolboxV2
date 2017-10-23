function vbhmm_plot_compact(hmm, imgfile, plotmode)
% vbhmm_plot_compact - compact plot of HMM
%
%   vbhmm_plot_compact(hmm, imgfile, plotmode)
%
% INPUTS
%   hmm      = hmm from vbhmm_learn
%   imgfile  = filename of image for plotting under the fixations (optional)
%   plotmode = 'r' - plot the transition matrix to the right of ROI plot (default)
%            = 'b' - plot the transition matrix below the ROI plot
%            = ''  - don't plot the transition matrix
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2016-11-22: ABC - created

[colors, colorfull] = get_color_list();

if nargin<2
  imgfile = '';
end
if nargin<3
  plotmode = 'r';
end

if ~isempty(imgfile)
  img = imread(imgfile);
  if ndims(img)==2
    img = cat(3,img,img,img);
  end
else
  img = [];
end

% get fixation plot dimensions
if ~isempty(img)
  xmin_img = 0;
  xmax_img = size(img,2);
  ymin_img = 0;
  ymax_img = size(img,1);
else
  error('not supported yet');
end

D = length(hmm.pdf{1}.mean);
K = size(hmm.trans,1);

% get location of transition matrix
switch(plotmode)
  case 'r'
    xmin_trans = xmax_img;
    xmax_trans = xmax_img + min(xmax_img, ymax_img);
    ymin_trans = ymin_img;
    ymax_trans = ymax_img;
    if (D>2)
      % set the location of the duration emissions
      xmin_dur = xmin_trans;
      xmax_dur = xmax_trans;
      ymin_dur = ymin_trans;
      ymax_dur = ymax_trans;
        
      % and move the transition matrix to the right
      w = (xmax_dur-xmin_dur);
      xmin_trans = xmin_trans + w;
      xmax_trans = xmax_trans + w;
    end
    
  case 'b'
    error('not supported yet');
end

%% plot the ROIs
plot_emissions([], [], hmm.pdf, img)
title('');

if isempty(plotmode)
  return
end
hold on

%% plot the duration emissions
if (D>2)
  % get size
  xw = xmax_dur - xmin_dur;
  yh = ymax_dur - ymin_dur;
  
  % get line for each pdf
  dur_z    = {};
  dur_t    = {};
  dur_mu_z = {};
  legs = {};
  for k=1:K
    mu     = hmm.pdf{k}.mean(3);
    sigma2 = hmm.pdf{k}.cov(3,3);
    ss = sqrt(sigma2);
    tmin = max(mu - 3*ss, 0);
    tmax = mu + 3*ss;
    t = linspace(tmin, tmax, 100);
    z = normpdf(t, mu, ss);
    
    dur_t{k}    = t;
    dur_z{k}    = z;
    dur_mu_z{k} = [mu, normpdf(mu,mu,ss)];
    
    legs{k} = sprintf('\\color{%s} %d: %d\\pm%d', colorfull{k}, k, round(mu), round(ss));
  end

  % find the t range and pt range
  tmin = 0;
  tmax = max(cat(2, dur_t{:}));
  zmin = 0;
  zmax = max(cat(2, dur_z{:}));

  % remap plot to canvas (image) coordinates
  padding = 20;
  yaxpadding = 65;
  textpadding = 5;
  textoffset = 20;
  t_map = @(t) ((t-tmin) / (tmax-tmin))*(xw-padding*2) + xmin_dur+padding;
  z_map = @(z) ymax_dur-yaxpadding - ((z-zmin) / (zmax-zmin))*(yh-yaxpadding);  % upside down
  
  % plot axes
  fill([xmin_dur+padding, xmin_dur+padding, xmax_dur-padding, xmax_dur-padding], ...
       [ymin_dur, ymax_dur-yaxpadding, ymax_dur-yaxpadding, ymin_dur], ...
       'w', 'linewidth', 0.5);
  
  % plot text values
  ytext = ymax_dur-yaxpadding+textpadding;
  text(xmin_dur+padding, ytext, sprintf('%d', floor(tmin)), ...
    'HorizontalAlignment', 'center', 'FontSize', 7, ...
    'VerticalAlignment', 'top');
  text(xmax_dur-padding, ytext, sprintf('%d', ceil(tmax)), ...
    'HorizontalAlignment', 'center', 'FontSize', 7, ...
    'VerticalAlignment', 'top');  
  
  % plot pdfs
  for k=1:K
    mytext = ytext+(k-1)*textoffset;
    plot(t_map(dur_t{k}), z_map(dur_z{k}),'Color', colors(k), 'linewidth', 2);
    
    % plot center line
    plot(t_map(dur_mu_z{k}(1))*[1 1], [mytext, z_map([dur_mu_z{k}(2)])], '--', 'Color', colors(k));
    
    % plot duration
    text(t_map(dur_mu_z{k}(1)), mytext, sprintf('%d', round(dur_mu_z{k}(1))), ...
      'HorizontalAlignment', 'center', 'FontSize', 7, ...
      'VerticalAlignment', 'top');
  end
  
  % plot labels
  for k=1:K 
    text(t_map(dur_mu_z{k}(1)), z_map(dur_mu_z{k}(2)/2), sprintf('%d', k), 'color', colors(k), ...
      'horizontalalignment', 'center');
  end
  
  % more compact legend
  text(xmax_dur-padding, ymin_dur, legs, 'color', 'black', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', ...
    'FontSize', 7, 'EdgeColor', 'black', 'Margin', 1);

  
end


%% plot the prior & transition matrix
padding=40;

% x spacing (common for both)
tx = linspace(xmin_trans+padding, xmax_trans, K+1);
tx = 0.5*(tx(2:end)+tx(1:end-1));

% overall y spacing
ty = linspace(ymin_trans+padding, ymax_trans, K+2);
ty = 0.5*(ty(2:end)+ty(1:end-1));

% y spacing for prior
typ = ty(1)-padding;

% y spacing for transition matrix
tyt = ty(2:end);

% cell size
if length(tyt)>1
  dy = (tyt(2) - tyt(1));
else
  dy = (ymax_trans-ymin_trans-padding);
end
if length(tx)>1
  dx = (tx(2) - tx(1));
else
  dx = (xmax_trans-xmin_trans-padding);
end

% plot prior (hack to trick matlab into plotting a vector)
%  ... make it plot a matrix instead.
imagesc(tx, typ+[-dy/4, dy/4], [hmm.prior(:)'; hmm.prior(:)'], [0 1]);

imagesc(tx, tyt, hmm.trans, [0 1]);
hold off
%colorbar

% plot probabilities
for j=1:K
  mycolor = getcolor(hmm.prior(j));
  text(tx(j),typ(1), sprintf('%.2f', hmm.prior(j)), ...
      'HorizontalAlignment', 'center', 'FontSize', 7, ...
      'Color', mycolor);
end

for j=1:K
  for k=1:K
    mycolor = getcolor(hmm.trans(j,k));
    text(tx(k),tyt(j), sprintf('%.2f', hmm.trans(j,k)), ...
      'HorizontalAlignment', 'center', 'FontSize', 7, ...
      'Color', mycolor);
  end
end

% plot ROI color strips
for j=1:K
  % labels for rows
  rectangle('Position', [xmin_trans+padding/4, tyt(j)-dy/2, padding*3/4, dy], 'FaceColor', colors(j));
  text(xmin_trans+padding*2.5/4, tyt(j), sprintf('%d', j), ...
    'HorizontalAlignment', 'center', 'FontSize', 7);
  
  % labels for columns
  rectangle('Position', [tx(j)-dx/2, tyt(1)-dy/2-padding*3/4, dx, padding*3/4], 'FaceColor', colors(j));
  text(tx(j), tyt(1)-dy/2-padding*1.5/4, sprintf('to %d', j), ...
    'HorizontalAlignment', 'center', 'FontSize', 7);
end

% prior
text(xmin_trans+padding*2.5/4, typ, 'prior', ...
  'Rotation', 90, 'FontSize', 7, 'HorizontalAlignment', 'center')

% reset axis
axis([min(xmin_img, xmin_trans), ...
      max(xmax_img, xmax_trans), ...
      min(ymin_img, ymin_trans), ...
      max(ymax_img, ymax_trans)]);
 
colormap gray;
%colorbar;

%plot_transprob(hmm.trans)
%plot_prior(hmm.prior)

function mycolor = getcolor(p)
if p > 0.3
  mycolor = 'k';
else
  mycolor = 'w';
end
