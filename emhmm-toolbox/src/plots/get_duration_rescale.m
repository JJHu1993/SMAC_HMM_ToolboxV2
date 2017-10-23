function myscale = get_duration_rescale(data)
% get_duration_rescale - get rescaling function for plotting duration as marker size
%
%    myscale = get_duration_rescale(data)
%
% returns an anonymous function, myscale
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


alldata = cat(1, data{:});


Tmin = min(alldata(:,3));
Tmax = max(alldata(:,3));

% rescale to a set size
szmin = 2;
szmax = 36;
myscale = @(x) [(x-Tmin)/Tmax*(szmax-szmin) + szmin];