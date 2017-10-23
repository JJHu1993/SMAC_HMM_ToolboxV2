function plot_transprob(trans)
% plot_transprob - plot transition matrix
%
%  plot_transprob(M)
%
% INPUT: 
%   M = transition matrix (from vbhmm)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

K = size(trans,1);
imagesc(trans, [0 1])
colorbar
set(gca, 'XTick', [1:K]);
set(gca, 'YTick', [1:K]);
for j=1:size(trans,1)
  for k=1:size(trans,2)
    if trans(j,k) > 0.3
      mycolor = 'k';
    else
      mycolor = 'w';
    end
    text(k,j, sprintf('%.2f', trans(j,k)), ...
      'HorizontalAlignment', 'center', 'FontSize', 7, ...
      'Color', mycolor);
  end
end
colormap default
title('transition matrix');
xlabel('to ROI');
ylabel('from ROI');
