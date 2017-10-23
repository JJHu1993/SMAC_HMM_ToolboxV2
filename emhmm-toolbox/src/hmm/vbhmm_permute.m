function hmm = vbhmm_permute(hmm_old, cl)
% vbhmm_permute - permute the state indices of an HMM
%
%   hmm = vbhmm_permute(hmm_old, cl)
%
% maps ROI "cl(i)" in hmm_old to ROI "i" in hmm
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong



%
% maps ROI for cl(i)-> i
%



hmm = hmm_old;

usegroups = isfield(hmm, 'group_ids');

if ~usegroups
  % without groups
  hmm.prior = hmm_old.prior(cl);
  hmm.trans = hmm_old.trans(cl,cl);
  if isfield(hmm_old, 'M')
    hmm.M = hmm_old.M(cl,cl);
  end
  if isfield(hmm_old, 'N1')
    hmm.N1 = hmm_old.N1(cl);
  end
else
  % with groups
  for g=1:length(hmm.prior)
    hmm.prior{g} = hmm_old.prior{g}(cl);
    hmm.trans{g} = hmm_old.trans{g}(cl,cl);
    hmm.M{g} = hmm_old.M{g}(cl,cl);
    hmm.N1{g} = hmm_old.N1{g}(cl);
    hmm.Ng{g} = hmm_old.Ng{g}(cl);
  end
end
  
if isfield(hmm_old, 'N')
  hmm.N = hmm_old.N(cl);
end
hmm.pdf = {hmm_old.pdf{cl}};

if isfield(hmm_old, 'gamma')
  for n=1:length(hmm.gamma)
    hmm.gamma{n} = hmm_old.gamma{n}(cl,:);
  end
end

if isfield(hmm, 'varpar')
  if ~usegroups
    hmm.varpar.epsilon = hmm_old.varpar.epsilon(cl, cl);
    hmm.varpar.alpha   = hmm_old.varpar.alpha(cl);
  else
    for g=1:length(hmm.prior)
      hmm.varpar.epsilon{g} = hmm_old.varpar.epsilon{g}(cl, cl);
      hmm.varpar.alpha{g}   = hmm_old.varpar.alpha{g}(cl);
    end
  end
  hmm.varpar.beta    = hmm_old.varpar.beta(cl);
  hmm.varpar.v       = hmm_old.varpar.v(cl);
  hmm.varpar.m       = hmm_old.varpar.m(:,cl);
  hmm.varpar.W       = hmm_old.varpar.W(:,:,cl);
end

