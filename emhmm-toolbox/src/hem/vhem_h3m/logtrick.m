function [s] = logtrick(lA)
% logtrick - "log sum trick" - calculate log(sum(A)) using only log(A) 
%
%   s = logtrick(lA)
%
%   lA = column vector of log values
%
%   if lA is a matrix, then the log sum is calculated over each column
% 
% ---
% H3M Toolbox 
% Copyright 2012, Emanuele Coviello, CALab, UCSD


[mv, mi] = max(lA, [], 1);
temp = lA - repmat(mv, size(lA,1), 1);
cterm = sum(exp(temp),1);
s = mv + log(cterm);


