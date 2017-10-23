function [gmm, Xpost, info] = gmm_learn(X, K, opt)
% gmm_learn - fit a GMM using the EM algorithm
%
%  [gmm, post, info] = gmm_learn(X, K, opt)
%
% Inputs:
%   X       = sample points: d x N (each column is a vector)
%   K       = number of components
%   opt.cvmode      = 'iid', 'diag', 'full'
%   opt.cvminreg    = minimum covariance regularization [default=1e-6]
%
%   opt.bkgndclass  = add a uniform background distribution to catch outliers
%                     using this value as the uniform likelihood.  Outlier
%                     points are ignored when computing the GMM parameters
%                     in the M-step.
%                     [default = 0, no background]
%                     the background component is the (K+1) column in the
%                     LLcomp and posterior (see gmm_ll).
%
%   opt.initmode    = 'one'    - one run of EM using initgmm
%                     'random' - trials with random initializations
%                     'split'  - component splitting [default]
%   opt.numtrials   = number of trials for random initialization [default=5]
%   opt.initgmm     = [1xK] component centers [default = use random points from X]
%                   = [1xN] cluster membership
%                   = initial gmm structure
%   opt.splitsched  = split schedule [default=1:K]
%   opt.splitLLterm = termination likelihood for splitting [default=1e-4]
%   opt.LLterm      = termination likelihood condition [default=1e-5]
%     
%   opt.groups     = make sure groups of vectors are clustered together
%                    [1 x N] vector with group IDs [default=[]]
%                    or {1 x G} vector array with groups
%
%   opt.showplot   = 0 - no plots [default]
%                    1 - show a final plot only
%                    2 - show a plot after each iteration
%                    3 - show a plot after each iteration (and pause)
%                    4 - show a pretty plot after each iteration (for saving)
%      .showplotfile = 'name_%05d' (fills in with iteration number)
%                      'name_%d_%%05d' (fills in K and iteration number(splitting mode))
%   opt.verbose    = 0 - at end
%                    1 - per iteration [default]
%                    2 - per iteration (one per line)
%
% OUTPUT
%  
%   gmm  = GMM model
%        .K      = number of components
%        .pi     = priors
%        .mu{}   = means
%        .cv{}   = covariance
%        .cvmode = 'iid', 'diag', 'full'
%        .bkgndclass = uniform background likelihood (0 = none)
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

%
% 2009-11-13 : added verbose=2 option
%            : split empty components with small prior
%
% 2012-03-07: ABC,Alvin - add bkgndclass

if ~isfield(opt, 'showplot');
  opt.showplot = 0;
end
if ~isfield(opt, 'LLterm')
  opt.LLterm = 1e-5;
end
if ~isfield(opt, 'cvminreg')
  opt.cvminreg = 1e-6;
end
if ~isfield(opt, 'verbose')
  opt.verbose = 1;
end
if ~isfield(opt, 'groups')
  opt.groups  = [];
end
if ~isfield(opt, 'bkgndclass')
  opt.bkgndclass = 0;
end

%%% convert groups format %%%
if ~isempty(opt.groups)
  if ~iscell(opt.groups)
    fprintf('converting groups format...\n');
    ug   = unique(opt.groups);
    gind = cell(1,length(ug));
    for i=1:length(ug)
      % find points in this group
      gind{i} = find(opt.groups == ug(i));
    end
    opt.groups = gind;
  end
end


maxiter = 1000;
showplot = opt.showplot;

[d, N] = size(X);

switch(opt.initmode)
 case 'one'
  %%% RUN EM ONCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  if (showplot)
    figem = figure;
  end
  
  %%% initialize! %%%
  if isfield(opt, 'initgmm') && isstruct(opt.initgmm)
    % initialize with GMM
    gmm = opt.initgmm;
    
  elseif isfield(opt, 'initgmm') && (length(opt.initgmm) == N)
    % initialize with cluster membership
    Y = zeros(N, K);
    for j=1:K
      ind = find(opt.initgmm==j);
      Y(ind,j) = 1;
    end
    if (sum(Y(:)) ~= N)
      error('bad initial cluster membership');
    end
            
    gmm.K      = K;
    gmm.cvmode = opt.cvmode;
    gmm.pi     = zeros(1,K);
    gmm.mu     = cell(1,K);
    gmm.cv     = cell(1,K);
    
    gmm = Mstep(gmm, X, Y, opt.cvminreg);
  
  else
    % initialize with centers
    if ~isfield(opt, 'initgmm') || isempty(opt.initgmm)
      % select K random points
      foo  = randperm(N);
    elseif (length(opt.initgmm) == K)
      % use specified centers
      foo  = opt.initgmm;
    else
      error('bad initgmm');
    end
      
    % initialize GMM
    gmm.K      = K;
    gmm.cvmode = opt.cvmode;
    gmm.pi     = ones(1,K)/K;
    cvall      = mean(var(X'))/4;  % average variance / 2
    for j=1:K
      gmm.mu{j} = X(:,foo(j));
      switch(opt.cvmode)
       case 'iid'
         gmm.cv{j} = cvall;
       case 'diag'
         gmm.cv{j} = cvall*ones(d,1);
       case 'full'
         gmm.cv{j} = cvall*eye(d);
        otherwise
         error('bad cvmode');
      end
    end
  end
  
  % add background class
  if (opt.bkgndclass  ~= 0)
    gmm.bkgndclass = opt.bkgndclass;
  end
  
  % initialize groups
  gind = opt.groups;
    
  %%% initialize EM %%%
  iter      = 1;
  dataLLall = zeros(1,maxiter);
  dataLL    = nan;
  
  %%% run EM %%%
  while(1)
    dataLLold = dataLL;
    
    %%% E-Step %%%
    [LL, LLcomp, Y] = gmm_ll(X, gmm);
    
    dataLL          = sum(LL);
    dataLLall(iter) = dataLL;
    
    %%% check convergence %%%
    dobreak = 0;
    if (iter>1)
      dLL = dataLL - dataLLold;
      pLL = abs(dLL / dataLL);
      if (opt.verbose)
        outstr = sprintf('* iter=%d: LL=%10.5g (dLL=%0.5g; pLL=%0.5g)      ', ...
          iter, dataLL, dLL, pLL);
        if (opt.verbose == 1)
          printline(outstr);
        elseif (opt.verbose > 1)
          fprintf(outstr);
          fprintf('\n');
        end
      end
      if (dLL < 0)
	warning('LL change is negative!');
      end
      if (pLL < opt.LLterm)
	dobreak = 1;
      end
    end
    if (iter > maxiter)
      warning('max iterations reached');
      dobreak = 1;
    end

    if showplot==4
      if ~isfield(opt, 'showplotfile')
        myplotname = '';
      elseif isempty(opt.showplotfile)
        myplotname = '';
      else
        myiter = iter;
        myplotname = sprintf(opt.showplotfile, myiter);
      end
      plotpretty(figem, gmm, X, dataLL, Y, myplotname);
    else
      if ((showplot > 1) || ((showplot==1) && dobreak))
        plotdebug(figem, gmm, X, (showplot==3));
      end
    end
    
    if (dobreak)
      if ~(opt.verbose)
	fprintf('* iter=%d: LL=%10.5g (dLL=%0.5g; pLL=%0.5g)', ...
		iter, dataLL, dLL, pLL);
      end
      break;
    end
    
    %%% M-Step %%%    
    gmm = Mstep(gmm, X, Y, opt.cvminreg, gind, LLcomp);
    
    % check for empty
    for j=1:gmm.K
      if isempty(gmm.cv{j})
	fprintf('empty cluster %d\n', j);
	gmm = dosplit(gmm, gmm.K, j);	
      end
    end
    
    
    iter = iter+1;
  end
    
  % output
  Xpost = Y;
  info.dataLL    = dataLL;
  info.dataLLall = dataLLall(1:iter);
  info.opt       = opt;
  
  fprintf('\n');
    
 case 'random'
  %%% RUN EM TRIALS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if ~isfield(opt, 'numtrials')
    opt.numtrials = 5;
  end
  
  for trial = 1:opt.numtrials
    myopt = opt;
    myopt.initmode = 'one';
    myopt.initgmm  = [];
    
    [tmpgmm, tmppost, tmpinfo] = gmm_learn(X, K, myopt);

    allgmm{trial}  = tmpgmm;
    allpost{trial} = tmppost;
    allinfo{trial} = tmpinfo;
    allLL(trial)   = tmpinfo.dataLL;
  end

  allLL
  
  % pick the best
  [foo, ind] = max(allLL)
  
  gmm   = allgmm{ind}
  Xpost = allpost{ind};
  info  = allinfo{ind};
  
 case 'split'
  %%% COMPONENT SPLITTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if ~isfield(opt, 'splitsched')
    opt.splitsched = 1:K;
  end
  if ~isfield(opt, 'splitLLterm')
    opt.splitLLterm = 1e-4;
  end
  
  % check schedule
  if (opt.splitsched(1) ~= 1)
    error('split schedule must start with 1');
  end
  if (opt.splitsched(end) ~= K)
    error('split schedule must end with K');
  end
  
  % for each K
  for j=1:length(opt.splitsched)
    myK = opt.splitsched(j);
    
    fprintf('*** K=%d ***\n', myK);
    
    if (j==1)
      % learn initial
      myopt = opt;
      myopt.initmode = 'one';
      % initialize by putting all points into the same cluster
      myopt.initgmm  = ones(1,N);   
      myopt.LLterm   = myopt.splitLLterm;
      
      % change the showplot name
      if isfield(myopt, 'showplotfile')
        if ~isempty(myopt.showplotfile)
          % fill in K
          myopt.showplotfile = sprintf(myopt.showplotfile, myK);
        end
      end
      
      [mygmm, mypost, myinfo] = gmm_learn(X, myK, myopt);
      
    else
      % do splitting
      mygmm = dosplit(mygmm, myK);
      
      if 0
        if (myK > 2*mygmm.K)
          error('split K too large');
        end
        
        % get the max eigenvalues of each component
        myeigs = zeros(1,mygmm.K);
        for jj=1:mygmm.K
          switch(mygmm.cvmode)
            case {'iid', 'diag'}
              myeigs(jj) = max(mygmm.cv{jj});
            case {'full'}
              myeigs(jj) = max(eig(mygmm.cv{jj}));
            otherwise
              error('bad cvmode');
          end
        end
        
        % split!
        while(mygmm.K < myK)
          % find largest eigs
          [foo, ind1] = max(myeigs);
          
          % target component
          ind2 = mygmm.K+1;
          
          fprintf('splitting %d to %d\n', ind1, ind2);
          
          % make a new component
          mygmm.K = mygmm.K+1;
          tmp = mygmm.pi(ind1);
          mygmm.pi(ind1) = tmp/2;
          mygmm.pi(ind2) = tmp/2;
          tmpcv = mygmm.cv{ind1};
          mygmm.cv{ind2} = tmpcv;
          tmpmu = mygmm.mu{ind1};
          
          % perturb mu in maximum variance direction
          % by (std/2)
          switch(mygmm.cvmode)
            case 'iid'
              pert = sqrt(tmpcv)/2;
              tmpd = pert*ones(d,1);
            case {'diag', 'full'}
              if strcmp(mygmm.cvmode, 'full')
                foo = diag(tmpcv);
              else
                foo = tmpcv;
              end
              [foo2, dind] = max(foo);
              pert = sqrt(foo2)/2;
              tmpd = zeros(d,1);
              tmpd(dind) = pert;
            otherwise
              error('bad mode');
          end
          
          mygmm.mu{ind1} = tmpmu-tmpd;
          mygmm.mu{ind2} = tmpmu+tmpd;
          
          % invalidate this cluster
          myeigs(ind1) = -1;
          
        end % end while myK
      end
      
      % run EM
      myopt = opt;
      myopt.initmode = 'one';
      myopt.initgmm  = mygmm;
      % use looser convergence when not K
      if (myK ~= K)
        myopt.LLterm   = opt.splitLLterm;
      end
      
      if isfield(myopt, 'showplotfile')
        if ~isempty(myopt.showplotfile)
          % fill in K
          myopt.showplotfile = sprintf(myopt.showplotfile, myK);
        end
      end
      
      [mygmm, mypost, myinfo] = gmm_learn(X, myK, myopt);
      
    end
    
    % save results for each split
    allgmm{j}  = mygmm;
    allinfo{j} = myinfo;
    allpost{j} = mypost;
  end
  
  gmm   = mygmm;
  Xpost = mypost;
  info  = myinfo;
  info.allgmm  = allgmm;
  info.allinfo = allinfo;
  info.allpost = allpost;
  
 otherwise
  error('bad initmode');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gmm = Mstep(gmm, X, Y, cvminreg, gind, LLcomp)
if (nargin<5)
  gind = [];
end
if (nargin<6)
  LLcomp = [];
end

% do group processing
if ~isempty(gind)
  Yold = Y;
  Y    = zeros(size(Y));
  
  % compute LL and posterior for each group
  for i=1:length(gind)
    mygind = gind{i};
    
    % compute LL
    gLL = sum(LLcomp(mygind,:), 1);
    
    % correct for too many gmm.pi terms (only need one!)
    gLL = gLL - (length(mygind)-1)*log(gmm.pi);


    % compute group posterior
    tmp = logtrick(gLL');
    gY = exp(gLL - tmp);
    
    % save in Y
    Y(mygind,:) = repmat(gY, length(mygind), 1);
  end
end


d    = size(X,1);
N    = size(X,2);
Nhat = sum(Y,1);
    
if isfield(gmm, 'bkgndclass') && (gmm.bkgndclass ~= 0)
  % ignore points in background class when calculating pi
  gmm.pi = Nhat(1:gmm.K) / sum(Nhat(1:gmm.K));
else
  % standard case
  gmm.pi = Nhat / N;
end

for j=1:gmm.K
  if (Nhat(j) < 0.1*1/N)
    mu = [];
    cv = [];
    
  else
    %Yj = repmat(Y(:,j)',d,1);    
    %mu = sum(Yj.*X,2) / Nhat(j);
    Yj = Y(:,j)';
    
    mu = sum(bsxfun(@times, Yj, X),2) / Nhat(j);
    
    %Xmu = X - repmat(mu,1,N);
    Xmu = bsxfun(@minus, X, mu);
    
    switch(gmm.cvmode)
     case 'iid'
      cv = sum(sum(bsxfun(@times, Yj, (Xmu.^2)),2)) / (d*Nhat(j));
      cv = max(cv, cvminreg);
     case 'diag'
      cv = sum(bsxfun(@times, Yj, (Xmu.^2)),2) / Nhat(j);
      cv = max(cv, cvminreg);
     case 'full'
      cv = bsxfun(@times,Yj,Xmu)*(Xmu') / Nhat(j);
      if (cvminreg > 0)
        [V, D] = eig(cv);
        Dd = diag(D);
        Dd = max(Dd, cvminreg);
        cv = V*diag(Dd)*V';
      end
     otherwise
      error('bad cvmode');
    end
  end
  
  gmm.mu{j} = mu;
  gmm.cv{j} = cv;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotdebug(figem, gmm, X, dopause)

d = size(X,1);

figure(figem)
if (d == 1)
  [hn,hx] = hist(X,100);
  bar(hx, hn/(sum(hn)*(hx(2)-hx(1))));
  gmm_plot1d(gmm, 'r');
else
  plot(X(1,:), X(2,:), '.b', 'MarkerSize', 2);
  gmm_plot2d(gmm, 'r');
end
drawnow
if (dopause)
  pause
end

function plotpretty(figem, gmm, X, dataLL, Y, myplotname)

d = size(X,1);
K = gmm.K;
figure(figem)
clf
% assume d==2

% Y*basecols = Xcol  --> [N x K] x [K x 3] --> [N x 3]
basecols{1} = [1 0 0];
basecols{2} = [1 0 0; 0 1 0];
basecols{3} = [1 0 0; 0 1 0; 0 0 1];
basecols{4} = [1 0 0; 0 1 0; 0 0 1; 1 1 0];
basecols{5} = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; ];
basecols{6} = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1];

Xcol = min(Y*basecols{K}, 1);

scatter(X(1,:), X(2,:), 10, Xcol, 'filled');
hold on
gmm_plot2d(gmm, 'k');

hold off

% set axis based on data
minx1 = min(X(1,:));
maxx1 = max(X(1,:));
minx2 = min(X(2,:));
maxx2 = max(X(2,:));
delx1 = maxx1-minx1;
delx2 = maxx2-minx2;
axis equal
axisx(minx1-delx1*0.2, maxx1+delx1*0.2);
axisy(minx2-delx2*0.2, maxx2+delx2*0.2);

grid on


% show data LL
title(sprintf('LL=%g', dataLL));

drawnow
pause
if ~isempty(myplotname)
  fprintf(' --- saving to %s ---\n', myplotname);
  savefig(myplotname);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if mytarget is present, split a component and replace mytarget
% otherwise, split mygmm until myK components
function mygmm = dosplit(mygmm, myK, mytarget)
if nargin<3
  mytarget = [];
end

if (myK > 2*mygmm.K)
  error('split K too large');
end
      
% get the max eigenvalues of each component
myeigs = zeros(1,mygmm.K);
for jj=1:mygmm.K
  if (isempty(mygmm.cv{jj}))
    myeigs(jj) = -1;
  else
    switch(mygmm.cvmode)
     case {'iid', 'diag'}
      myeigs(jj) = max(mygmm.cv{jj});
     case {'full'}
      myeigs(jj) = max(eig(mygmm.cv{jj}));
     otherwise
      error('bad cvmode');
    end
  end
end
      
if ~isempty(mytarget)
  dotarget = 1;
else
  dotarget = 0;
end  

% split!
while (mygmm.K < myK) || dotarget
  % find largest eigs
  [foo, ind1] = max(myeigs);
	
  % target component
  if (dotarget)
    ind2 = mytarget;
    dotarget = 0;
  else
    ind2 = mygmm.K+1;
    mygmm.K = mygmm.K+1;
  end
  
  fprintf('splitting %d to %d\n', ind1, ind2);
	
  % make a new component
  tmp = mygmm.pi(ind1);
  mygmm.pi(ind1) = tmp/2;
  mygmm.pi(ind2) = tmp/2;
  tmpcv = mygmm.cv{ind1};
  mygmm.cv{ind2} = tmpcv;
  tmpmu = mygmm.mu{ind1};
	
  % perturb mu in maximum variance direction
  % by (std/2)
  switch(mygmm.cvmode)
   case 'iid'
    pert = sqrt(tmpcv)/2;
    d = size(mygmm.mu{1},1);
    tmpd = pert*ones(d,1);
   case 'diag'
    [foo2, dind] = max(tmpcv);
    pert = sqrt(foo2)/2;
    tmpd = zeros(length(tmpmu),1);
    tmpd(dind) = pert;	 
   case 'full'
    [tV tD] = eig(tmpcv);
    [foo, ti] = max(diag(tD));
    tmpd = tV(:,ti) * (sqrt(foo)/2);
    
   otherwise
    error('bad mode');
  end
	
  mygmm.mu{ind1} = tmpmu-tmpd;
  mygmm.mu{ind2} = tmpmu+tmpd;
	
  % invalidate this cluster
  myeigs(ind1) = -1;
  
end % end while myK

