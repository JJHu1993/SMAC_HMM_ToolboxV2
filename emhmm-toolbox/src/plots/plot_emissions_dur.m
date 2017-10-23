function plot_emissions_dur(data, gamma, pdf)
% plot_emissions_dur - plot emission densities for durations
%
%     plot_emissions_dur(data, gamma, pdf)
%
%  INPUTS
%    data  = data (same format as used by vbhmm_learn)
%    gamma = assignment variables for each data point to emission density
%    pdf   = the emission densities (from vbhmm)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


% VERSIONS
% 2017-01-17: ABC - iniital version

[color, colorfull] = get_color_list();

hold on

maxzs = [];
maxt = 0;
K = length(pdf);
legs = {};
for k = 1:K
  mu    = pdf{k}.mean(3);
  sigma2 = pdf{k}.cov(3,3);
  [t, z] = plot1D(mu,sigma2,color(k));
  maxzs(k) = max(z);
  maxt = max(max(t), maxt);
  legs{k} = sprintf('\\color{%s} %d: %d\\pm%d', colorfull{k}, k, round(mu), round(sqrt(sigma2)));
end
maxz = max(maxzs);

% plot data
if ~isempty(data)
  out = cat(1, data{:});
  gamma = cat(2, gamma{:});
  [~,cl] = max(gamma,[],1);
  K = size(gamma,1);
  
  D = size(data{1}, 2);
  if D <= 2
    error('data has no duration (dimension 3)');    
  end

  yoffset = maxz/10;
  yrand   = yoffset/4;
  for k=1:K
    ii = find(cl==k);
    % randomly y-offset the data
    randoffset = yrand*(2*rand(1,length(ii))-1);
    scatter(out(ii,3), -yoffset*(k-1)*ones(1,length(ii)) + randoffset, 6, [color(k) 'o']);
  end    
  
  ymin = -yoffset*((k-1)+0.5);
else
  ymin = 0;
end

% plot labels
for k = 1:K
  mu = pdf{k}.mean(3);
  text(mu, maxzs(k)/2, sprintf('%d', k), 'color', color(k), ...
    'horizontalalignment', 'center');
end

% get axes
ymax = maxz;
xmin = 0;
xmax = maxt;

% plot axes
plot([xmin, xmax], [0, 0], 'k-');

% compact legend
text(xmax, ymax, legs, 'color', 'black', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', ...
    'FontSize', 7, 'EdgeColor', 'black', 'Margin', 1);

grid on
hold off
title('ROIs (duration)');
xlabel('t');
ylabel('p(t)');

% reset axes
axis([xmin, xmax, ymin, ymax]);


%% plot a Gaussian curve
function [t, z, h] = plot1D(mu,sigma2,color)

ss = sqrt(sigma2);

tmin = mu - 3*ss;
tmax = mu + 3*ss;
t = linspace(tmin, tmax, 100);
z = normpdf(t, mu, ss);

h = plot(t, z,'Color', color, 'linewidth', 2);
plot([mu, mu], [0, normpdf(mu,mu,ss)], '--', 'Color', color);
