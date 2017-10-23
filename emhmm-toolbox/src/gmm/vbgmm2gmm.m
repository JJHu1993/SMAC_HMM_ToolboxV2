function [gmm, post, info] = vbgmm2gmm(vbgmm, sortclusters)
% vbgmm2gmm - convert vbgmm structure to gmm structure
%
%  [gmm, post, info] = vbgmm2gmm(vbgmm, sortclusters)
%
% GMM structure
%        .K      = number of components
%        .pi     = priors
%        .mu{}   = means
%        .cv{}   = covariance
%        .cvmode = 'iid', 'diag', 'full'
%        .bkgndclass = uniform background likelihood (0 = none)
%
% VBGMM structure
%   out.m = m;
%   out.C = C;
%   out.weight = alpha/sum(alpha);
%   out.LL = L;
%   out.iter = iter;
%   out.PriorPar = PriorPar;
%   out.initmix  = mix;
%   out.r = r;
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

if nargin<2
  sortclusters = '';
end


gmm.K  = length(vbgmm.weight);
gmm.pi = vbgmm.weight;
gmm.mu = cell(1,gmm.K);
gmm.cv = cell(1,gmm.K);
for j=1:gmm.K
  gmm.mu{j} = vbgmm.m(:,j);
  gmm.cv{j} = vbgmm.C(:,:,j);
end
gmm.cvmode = 'full';
gmm.bkgndclass = 0;

post = vbgmm.r;

info.vbgmm = vbgmm;


% post processing
if ~isempty(sortclusters)
  % reorder based on cluster size
  if (sortclusters == 'd')
    gmmold = gmm;
    postold = post;
    [pisort, pii] = sort(gmmold.pi(:), 1, 'descend');
    
    % reorder posterior matrix
    post = postold(:,pii);
    
    % reorder GMM components
    for q=1:gmmold.K
      nq = pii(q);
      gmm.pi(q) = gmmold.pi(nq);
      gmm.mu{q} = gmmold.mu{nq};
      gmm.cv{q} = gmmold.cv{nq};
    end
  end
end
