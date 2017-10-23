function [group_hmms] = h3m_to_hmms(h3m, covmode)
% h3m_to_hmms - convert an H3M into a list of hmms
%
%     [group_hmms, info] = h3m_to_hmms(h3m, covmode)
%
% INPUT:  h3m        = H3M from H3M toolbox
%         covmode    = type of covariance
% OUTPUT: group_hmms = set of HMMs (using vbhmm format)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


K = h3m.K;

hmms = cell(1,K);

% convert each H3M component into an HMM
for j=1:K
  myhmm = h3m.hmm{j};
  clear tmphmm
  
  % state prior, transition matrix
  tmphmm.prior = myhmm.prior;
  tmphmm.trans = myhmm.A;

  % for each emission density...
  for i = 1:length(myhmm.emit)
    tmphmm.pdf{i}.mean = myhmm.emit{i}.centres;       
    switch(myhmm.emit{i}.covar_type)
      case 'diag'
        tmphmm.pdf{i}.cov = diag(myhmm.emit{i}.covars);
      case 'full'
        tmphmm.pdf{i}.cov = myhmm.emit{i}.covars;
      otherwise
        error('bad covmode');
    end    
  end
  
  hmms{j} = tmphmm;
end

% assignment info
group_hmms.Z     = h3m.Z;
group_hmms.LogLs = h3m.LogLs;
group_hmms.LogL  = h3m.LogL;

% get cluster assignments
[foo, maxZ] = max(group_hmms.Z, [], 2);
group_hmms.label = maxZ(:)';

% get cluster memberships
group_hmms.groups = {};
group_hmms.group_size = [];
for j=1:length(hmms)
  group_hmms.groups{j} = find(group_hmms.label == j);
  group_hmms.group_size(j) = length(group_hmms.groups{j});
end

% clusters
group_hmms.hmms = hmms;



