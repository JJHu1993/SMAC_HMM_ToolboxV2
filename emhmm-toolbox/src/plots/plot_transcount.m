function plot_transcount(M)
% plot_transcount - plot transition counts
%
%  plot_transcount(M)
%
% INPUT: 
%   M = transition count matrix (from vbhmm)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

maxM = max(M(:));
K = size(M,1);
imagesc(M);
colorbar
set(gca, 'XTick', [1:K]);
set(gca, 'YTick', [1:K]);
for j=1:size(M,1)
  for k=1:size(M,2)
    if (M(j,k) > 0.3*maxM)
      mycolor = 'k';
    else
      mycolor = 'w';
    end
    text(k,j, sprintf('%0.1f', M(j,k)), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', mycolor);
  end
end
colormap gray
title('transition counts');
xlabel('to ROI');
ylabel('from ROI');
