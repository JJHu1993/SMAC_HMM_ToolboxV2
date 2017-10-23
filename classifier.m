function [stats, success_rate] = classifier(categoric_var, gaze_descriptors,classifier_type,cross_validation)
%Inputs:
%categoric_var= cell containing experimental condition names (strings)
%gaze_descriptors = cell containing gaze descriptors x experimental conditions.
%Gaze descriptors are computed with Compute_HMM_descriptors, normalized and regularized.
%They contain priors, transition matrix coefficients, state center coordinates, state covariance
%coefficients along the x and y axes).(observers x hmm parameters)
%classifier_type = name of the classifier used (string)

%Outputs:
%success_success_rate = number of correctly classified scanpaths / total number of scanpaths.
%stats = MANOVA stats structure

stats=NaN;
success_rate=NaN;
%Initialize variables
oksuccess_rate=0;
tested_obs=0;
all_cat=[];
all_obs=[];
for icat=1:length(categoric_var)%create categorical vector
    one_obs_feat=gaze_descriptors{1,icat};
    all_obs=[all_obs;one_obs_feat]; % Gaze descriptors sorted by classes
    
    cat_cell=cell(size(one_obs_feat,1),1);
    cat_cell(:)={char(categoric_var(icat))};
    all_cat=[all_cat; cat_cell]; % Class labels
end

%MANOVA (equivalent to LDA but this function gives the Eigen vectors)
[~,~,stats] =manova1(all_obs,all_cat);

% k-fold cross-validation
if cross_validation==1
    c = cvpartition(all_cat,'LeaveOut');%leave-one-out
else
    c = cvpartition(all_cat,'k',cross_validation);
end

switch classifier_type
    case 'LDA'
        fun = @(xT,yT,xt,yt)(sum(strcmp(yt,classify(xt,xT,yT,'linear'))));
        rate = crossval(fun,all_obs,all_cat,'partition',c);
        success_rate=sum(rate)/sum(c.TestSize);
    case 'diaglinear'
        fun = @(xT,yT,xt,yt)(sum(strcmp(yt,classify(xt,xT,yT,'diaglinear'))));
        rate = crossval(fun,all_obs,all_cat,'partition',c);
        success_rate=sum(rate)/sum(c.TestSize);
    case 'QDA'
        fun = @(xT,yT,xt,yt)(sum(strcmp(yt,classify(xt,xT,yT,'quadratic'))));
        rate = crossval(fun,all_obs,all_cat,'partition',c);
        success_rate=sum(rate)/sum(c.TestSize);
    case 'diagquadratic'
        fun = @(xT,yT,xt,yt)(sum(strcmp(yt,classify(xt,xT,yT,'diagquadratic'))));
        rate = crossval(fun,all_obs,all_cat,'partition',c);
        success_rate=sum(rate)/sum(c.TestSize);
    case 'mahalanobis'
        fun = @(xT,yT,xt,yt)(sum(strcmp(yt,classify(xt,xT,yT,'mahalanobis'))));
        rate = crossval(fun,all_obs,all_cat,'partition',c);
        success_rate=sum(rate)/sum(c.TestSize);
    case 'AdaBoostBinary'
        adaStump = fitensemble(all_obs,all_cat,'AdaBoostM1',50,'Tree');
        if cross_validation==1
            cvada = crossval(adaStump,'LeaveOut','on');%leave-one-out
        else
            cvada = crossval(adaStump,'KFold',cross_validation);
        end
        success_rate = 1-   kfoldLoss(cvada);
        
    case 'AdaBoostMultiClass'
        adaStump = fitensemble(all_obs,all_cat,'AdaBoostM2',50,'Tree');
        if cross_validation==1
            cvada = crossval(adaStump,'LeaveOut','on');%leave-one-out
        else
            cvada = crossval(adaStump,'KFold',cross_validation);
        end
        success_rate = 1-   kfoldLoss(cvada);
        
    case 'RandomForest'
        Random_Forest_Stump = fitensemble(all_obs,all_cat,'Bag',50,'Tree','Type','classification');
        if cross_validation==1
            cvrf = crossval(Random_Forest_Stump,'LeaveOut','on');%leave-one-out
        else
            cvrf = crossval(Random_Forest_Stump,'KFold',cross_validation);
        end
        success_rate = 1-   kfoldLoss(cvrf);
        
    case 'SVMBinary'
        SVMModel = fitcsvm(all_obs,all_cat,'Standardize',true,'ClassNames',categoric_var);
        if cross_validation==1
            cvsvm = crossval(SVMModel,'LeaveOut','on');%leave-one-out
        else
            cvsvm = crossval(SVMModel,'KFold',cross_validation);
        end
        success_rate = 1-   kfoldLoss(cvsvm);
        
    case 'SVMMultiClass'
        Mdl = fitcecoc(all_obs,all_cat);
        if cross_validation==1
            cvsvm = crossval(Mdl,'LeaveOut','on');%leave-one-out
        else
            cvsvm = crossval(Mdl,'KFold',cross_validation);
        end
        success_rate = 1-   kfoldLoss(cvsvm);
end

%  case 'RVM'
%      if length(categoric_var)>2
%          error('Multiclass RVM not implemented yet. Please only use RVM for 2-class problems')
%      end
%      addpath './PRML_functions'
%      pos=ones(size(gaze_descriptors{1,1},1),1);
%      neg=zeros(size(gaze_descriptors{1,2},1),1);
%      s_all_cat=[pos; neg];
%      s_all_cat(isub)=[];
%
%      [model, ~] = rvmBinEm(s_all_obs',s_all_cat');
%      [classestimate, ~] = rvmBinPred(model,observation');
%
%      if classestimate==1
%          class=categoric_var(1);
%      elseif classestimate==0
%          class=categoric_var(2);
%      end
end