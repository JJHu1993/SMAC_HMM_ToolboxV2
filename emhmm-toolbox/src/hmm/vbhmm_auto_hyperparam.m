function vbopt = vbhmm_auto_hyperparam(vbopt, data, img, opt)
% vbhmm_auto_hyperparam - automatically set some hyperparameters for vbhmm_learn using the data
%
%    vbopt = vbhmm_auto_hyperparam(vbopt, data, img, opt)
%
% INPUTS
%     vbopt = existing vbopt structure (or [])
%      data = data to be used for learning the hmm.  same format as for vbhmm_learn
%             or cell array of data.
%       img = the image (used to get the center of image), or filename for image.
%
%       opt = options for setting hyperparameters:
%             'c' - set mu as the center of the image, and duration as 250ms (if needed).
%                   set W so that the prior ROI width is 1/8 of the image width, 
%                      and duration range is 100m, i.e, std is 25ms.
%             'd' - data-driven method to set the hyperparameters (Empirical Bayes)
%                   set mu as the mean fixation location (and duration) according to the data.
%                   set W using the standard deviation of the data (assumes the ROI is circular)
%       
% OUTPUTS
%     vbopt = vbopt structure with specified hyperparameters set.
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% VERSIONS
%  2017-01-18 - ABC - initial version


if isempty(vbopt)
  vbopt = struct;
end

if isstr(img)
  img = imread(img);
end

alldata = cat(1, data{:});
if iscell(alldata)
  alldata = cat(1, alldata{:});
end
D = size(alldata, 2);

switch(opt)
  case 'c'
    % center of image
    vbopt.mu = 0.5*[size(img,2); size(img,1)];
    if (D==3)
      % 250 ms fixation duration
      vbopt.mu(3) = 250;
    end
    
    % ROI (width of 4 standard deviations) is 1/8 of size of image
    width = 0.5*(size(img,1) + size(img,2));
    s = (width / 8) / 4;
    
    if (D == 2)
      vbopt.W = s.^(-2);
    else
      % 100ms duration range (std=25)
      st = 25;
      % assume ROI is circular
      vbopt.W = [s, s, st].^(-2);
    end
          
  case 'd'
    % mean fixation location and mean duration
    vbopt.mu = mean(alldata, 1)';

    % get std of fixation location (assume circular)
    vxy = var(alldata(:,1:2));
    s = sqrt(mean(vxy));
      
    if (D == 2)
      vbopt.W = s.^(-2);
    else
      st = std(alldata(:,3));
      
      % assume ROI is circular
      vbopt.W = [s, s, st].^(-2);
    end

  otherwise
    error('unknown option');
end







