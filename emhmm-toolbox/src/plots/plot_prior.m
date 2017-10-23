function plot_prior(prior)
% plot_prior - plot fixations on an image
%
%  plot_prior(prior)
%
% INPUT: 
%   prior - prior distribution
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


color = get_color_list();
set(gca, 'XTick', [1:length(prior)]);
hold on
for i=1:length(prior)
  bar(i, prior(i), color(i));
end
grid on
title('prior');
ylabel('probability');
xlabel('ROI');
