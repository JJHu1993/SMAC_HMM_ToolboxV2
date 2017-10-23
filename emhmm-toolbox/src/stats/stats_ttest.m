function [p, info, lld] = stats_ttest(hmms1, hmms2, data1, opt)
% stats_ttest - run t-test to compare two HMMs
%
%  [p, info] = stats_ttest(hmms1, hmms2, data1, opt)
%
% INPUTS
%   hmms1 = 1st set of HMM (a cell array of HMMS, one for each subject)
%   hmms2 = 2nd set of HMM (a cell array of HMMS, one for each subject)
%   data1 = cell array of data associated with each subject in hmms1
%           data1{i}{j} - for each subject i (belonging to hmms1{i}), and j-th trial
%    opt = 'n' - normalize log-likelihood of each sequence by the length of the sequence (default)
%           '' - no special options
%
% NOTE
%   if hmms1 is a single hmm, then the same hmms1 will be used for each data1{i}.  
%   Likewise for hmms2.
%
% OUTPUTS
%      p = the p-value
%   info = other info from the t-test
%    lld = the log-likelihood differences used for the t-test
%
% SEE ALSO
%   vbhmm_kld
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

if nargin<4
  opt = 'n';
end

N = length(data1);
allLL1 = zeros(1,N);
allLL2 = zeros(1,N);

% for each subject
for i=1:N
  % get the HMMs
  if iscell(hmms1)
    myhmm1 = hmms1{i};
  else
    myhmm1 = hmms1;
  end
  if iscell(hmms2)
    myhmm2 = hmms2{i};
  else
    myhmm2 = hmms2;
  end
  
  % calculate log-likelihoods under each HMM
  ll1 = vbhmm_ll(myhmm1, data1{i}, opt);
  ll2 = vbhmm_ll(myhmm2, data1{i}, opt);
  
  % calculate the mean log-likelihood for this subject
  allLL1(i) = mean(ll1);
  allLL2(i) = mean(ll2);
end

% do a t-test between the mean log-likelihoods of all subjects
% theoretically, ll1 >= ll2, since hmm1 was learned from data1.
% in other words, (ll1-ll2) >= 0.  
% so we can use a right-tailed test to check if the mean is > 0.
[h,p,ci,stats] = ttest(allLL1, allLL2, 'tail', 'right');

% tim's version
% [h,p,ci,stats] = ttest(allLL1, allLL2, 0.05, 'right');


% output results
lld = allLL1 - allLL2;
info = stats;
info.ci = ci;


