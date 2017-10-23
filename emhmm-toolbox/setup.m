% setup the path for the toolbox
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


myname = mfilename('fullpath');

[pathstr,name,ext] = fileparts(myname);

gdir = [pathstr filesep 'src'];
ddir = [pathstr filesep 'demo'];

addpath(genpath(gdir))

