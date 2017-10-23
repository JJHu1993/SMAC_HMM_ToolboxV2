function hmm_new = vbhmm_standardize(hmm, mode)
% vbhmm_standardize - standardize an HMM's states (ROIs) to be consistent.
%
%   hmm_new = vbhmm_standardize(hmm, mode)
%
%     hmm = an HMM from vbhmm_learn
%           or a group HMM from vhem_cluster
%
%    mode = 'e' - sort by emission frequency (overall number of fixations in an ROI)
%         = 'p' - sort by prior frequency (number of first-fixations in an ROI)
%         = 'f' - sort by most-likely fixation path
%                 (state 1 is most likely first fixation. State 2 is most likely 2nd fixation, etc)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% 2016-11-22: ABC - created

% TODO: sort by emission Gaussian spatial size
%       sort by emission Gaussian location

% run on each hmm
if isfield(hmm, 'hmms')
  hmm_new = hmm;
  for i=1:length(hmm.hmms)
    hmm_new.hmms{i} = vbhmm_standardize(hmm.hmms{i}, mode);
  end
  
  return
end

switch(mode)
  % reorder based on cluster size
  case {'d', 'e'}
    if mode=='d'
      warning('standardization mode ''d'' is deprecated. Use ''e''');
    end
    
    if isfield(hmm, 'N')
      [wsort, wi] = sort(hmm.N, 1, 'descend');
    else
      error('cluster size unknown - not from vbhmm_learn');
    end
    
  % sort by prior
  case 'p'
    [wsort, wi] = sort(hmm.prior(:), 1, 'descend');

  % sort by likely fixation path
  case 'f'
    % find starting point
    A = hmm.trans;
    for t=1:length(hmm.prior)
      % get next most-likely fixation
      if (t==1)
        [~, curf] = max(hmm.prior);
      else
        [~, curf] = max(A(curf,:));
      end        
      wi(t) = curf;
      
      % invalidate this fixation
      A(:,curf) = -1;
    end
    
  otherwise
    error('unknown mode');
end


% permute states
hmm_new = vbhmm_permute(hmm, wi);