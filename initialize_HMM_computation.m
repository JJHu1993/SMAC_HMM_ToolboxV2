function vbopt=initialize_HMM_computation(im)
% Initialize variational HMM computation parameters

%vbopt = structure containing other options as below:
%   VB hyper-parameters:
%     vbopt.alpha = Dirichlet distribution concentration parameter -- large value
%                   encourages uniform prior, small value encourages concentrated prior (default=0.1)
%                   Another way to think of it is in terms of "virtual" samples -- A typical way to 
%                   estimate the probability of something is to count the number of samples that it
%                   occurs and then divide by the total number of samples, 
%                   i.e. # times it occurred / # samples.  
%                   The alpha parameter of the Dirichlet adds a virtual sample to this estimate, 
%                   so that the probability estimate is (# times it occurred + alpha) / # samples.  
%                   So for small alpha, then the model will basically just do what the data says.  
%                   For large alpha, it forces all the probabilities to be very similar (i.e., uniform).
%     vbopt.mu    = prior mean - it should be dimension D;
%                   for D=2 (fixation location), default=[256;192] -- typically it should be at the center of the image.
%                   for D=3 (location & duration), default=[256,192,250]
%     vbopt.W     = size of the inverse Wishart distribution (default=0.005).
%                   If a scalar, W is assumed to be isotropic: W = vbopt.W*I.
%                   If a vector, then W is a diagonal matrix:  W = diag(vbopt.W);
%                   This determines the covariance of the ROI prior.
%                   For example W=0.005 --> covariance of ROI is 200 --> ROI has standard devation of 14.
%                   
%     vbopt.v     = dof of inverse Wishart, v > D-1. (default=10) -- larger values give preference to diagonal covariance matrices.
%     vbopt.beta  = Wishart concentration parameter -- large value encourages use of
%                   prior mean & W, while small value ignores them. (default=1)
%     vbopt.epsilon = Dirichlet distribution for rows of the transition matrix (default=0.1). 
%                     The meaning is similar to alpha, but for the transition probabilities.
%  
%   EM Algorithm parameters
%     vbopt.initmode     = initialization method (default='random')
%                            'random' - initialize emissions using GMM with random initialization (see vbopt.numtrials)
%                            'initgmm' - specify a GMM for the emissions (see vbopt.initgmm)
%                            'split' - initialize emissions using GMM estimated with component-splitting
%     vbopt.numtrials    = number of trails for 'random' initialization (default=50)
%     vbopt.random_gmm_opt = for 'random' initmode, cell array of options for running "gmdistribution.fit".
%                            The cell array should contain pairs of the option name and value, which are recognized
%                            by "gmdistribution.fit".
%                            For example, {'CovType','diagonal','SharedCov',true,'Regularize', 0.0001}.
%                            This option is helpful if the data is ill-conditioned for the standard GMM to fit.
%                            The default is {}, which does not pass any options.
%     vbopt.initgmm      = initial GMM for 'initgmm':
%                            initgmm.mean{k} = [1 x K]
%                            initgmm.cov{k}  = [K x K]
%                            initgmm.prior   = [1 x K]                               
%     vbopt.maxIter      = max number of iterations (default=100)
%     vbopt.minDiff      = tolerence for convergence (default=1e-5)
%     vbopt.showplot     = show plots (default=1)
%     vbopt.sortclusters = '' - no sorting [default]
%                          'e' - sort ROIs by emission probabilites
%                          'p' - sort ROIs by prior probability
%                          'f' - sort ROIs by most-likely fixation path [default]
%                              (see vbhmm_standardize for more options)
%     vbopt.groups       = [N x 1] vector: each element is the group index for a sequence.
%                          each group learns a separate transition/prior, and all group share the same ROIs.
%                          default = [], which means no grouping used
%     cvopt.fix_cov      = fix the covariance matrices of the ROIs to the specified matrix.
%                          if specified, the covariance will not be estimated from the data.
%                          The default is [], which means learn the covariance matrices.
%     vbopt.fix_clusters = 1 - keep Gaussian clusters fixed (don't learn the Gaussians)
%                          0 - learn the Gaussians [default]
%     vbopt.
%
%     vbopt.verbose      = 0 - no messages
%                        = 1 - a few messages showing progress [default]
%                        = 2 - more messages
%                        = 3 - debugging


%     vbopt.alpha = 1e-25;
%     vbopt.epsilon = 1e-25;
%     vbopt.W     = 0.0005;
%     vbopt.beta  = 1;
%     vbopt.v     = 10;
%     vbopt.initmode = 'random';
%     vbopt.numtrials = 5;
%     vbopt.sortclusters = 'd';

vbopt.mu  = [size(im,2)/2;size(im,1)/2];%center of the scene

vbopt.showplot = 0;

vbopt.verbose=0;

%constrain covariance to fixed matrix,
%vbopt.fix_cov=diag([500 500]);
vbopt.fix_cov=[];


end