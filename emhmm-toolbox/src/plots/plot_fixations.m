function plot_fixations(data, img, LL)
% plot_fixations - plot fixations on an image
%
%  plot_fixations(data, img, LL)
%
% INPUT: 
%   data  - data (same format as used by vbhmm_learn)
%   img   - image for background
%   LL    - the log-likelihood (for display)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% VERSIONS
% 2017-01-17: ABC - add duration as marker size.


if ~isempty(data)
  D = size(data{1}, 2);
  if D > 2
    myscale = get_duration_rescale(data);
  end
end

if ~isempty(img)
  imshow(img);
end
axis ij
hold on
for i=1:length(data)
  if (D > 2)
    % plot duration as marker size
    szs1 = myscale(data{i}(1,3));
    szs2 = myscale(data{i}(2:end,3));
  else
    % fixed marker size
    szs1 = 6;
    szs2 = 6;
  end
  % plot x,y location, and duration as marker size
  plot(data{i}(:,1), data{i}(:,2), '-');
  scatter(data{i}(1,1), data{i}(1,2), szs1, 'o'); % first fixation
  scatter(data{i}(2:end,1), data{i}(2:end,2), szs2, 'o'); % others
end
hold off
title(sprintf('fixations (LL=%g)', LL));
