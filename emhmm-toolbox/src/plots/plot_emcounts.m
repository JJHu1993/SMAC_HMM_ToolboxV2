function plot_emcounts(N)
% plot_emcounts - plot emission counts
%
%  plot_emcounts(N) 
%
% INPUTS
%   N = emmission histogram
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

color = get_color_list();
set(gca, 'XTick', [1:length(N)]);
hold on
for i=1:length(N)
  bar(i,N(i),color(i));
end
title('ROI counts');
grid on
ylabel('count');
xlabel('ROI');