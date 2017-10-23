%If you use this toolbox, please cite Coutrot et al., 
%"Scanpath modeling and classification with Hidden Markov Models", Behavior
%Research Methods, 2017
% Classify tasks from Coutrot's dataset (conversational videos, 2 auditory conditions) with HMM-based gaze descriptors
clear
close all
load HMM_descriptor_Coutrot %computed with exemple_Compute_HMM_descriptors.m
all_stim=fieldnames(example_HMM_descriptor_Coutrot);

%LDA 1st Eigen Vector
%ini_stim=char(all_stim(1));
%ini_descriptor=example_HMM_descriptor_Coutrot.(ini_stim).with_os.gaze_descriptor;
%LDA_1st_Eigen_vect=NaN(size(ini_descriptor,2),length(all_stim));

%Correct classification vector
correct_classif=NaN(length(all_stim),1);

% For each stimulus, a correct classification score is computed
for istim=1:length(all_stim)
    
    im_name=char(all_stim(istim));
    
    %% Load HMM-based gaze descriptors from HMM_descriptor_Coutrot.mat
    gaze_descriptor_ws=example_HMM_descriptor_Coutrot.(im_name).with_os.gaze_descriptor;
    gaze_descriptor_wos=example_HMM_descriptor_Coutrot.(im_name).without_os.gaze_descriptor;
    
    %% Normalization to zero mean and unit std
    norm_gaze_descriptor_ws=zscore(gaze_descriptor_ws')';
    norm_gaze_descriptor_wos=zscore(gaze_descriptor_wos')';
    
    %% Regularization
    all_task = [norm_gaze_descriptor_ws;norm_gaze_descriptor_wos];
    lambda_I_all=0.00001*eye(size(all_task));
    regul_gaze_descriptor_all = all_task - lambda_I_all.*all_task + lambda_I_all ;
    regul_gaze_descriptor_ws=regul_gaze_descriptor_all(1:size(norm_gaze_descriptor_ws,1),:);
    regul_gaze_descriptor_wos=regul_gaze_descriptor_all(size(norm_gaze_descriptor_ws,1)+1:...
        size(norm_gaze_descriptor_ws,1)+size(norm_gaze_descriptor_wos,1),:);
    
    
    %% Choose classes to classify
    gaze_descriptors={regul_gaze_descriptor_ws, regul_gaze_descriptor_wos};
    categoric_var={'with_os', 'without_os'};
    
    %% Select type of classifier
    classifier_type='LDA';
    %  classifier_type='diagquadratic';
    % classifier_type='mahalanobis';
    %classifier_type='SVMBinary';
    %classifier_type='SVMMultiClass';
    %classifier_type='AdaBoostBinary';
    %classifier_type='AdaBoostMultiClass';
    % classifier_type='RVM';%Only for 2-class problems
    % classifier_type='AdaBoost';%Only for 2-class problems
    % classifier_type= 'RandomForest';
    
    %% k-fold cross-validation
  cross_validation=1;
% % if cross_validation==1
% %     leave-one-out
% % else
% %    'k'-cross_validation
% % end
    try
        [lda_stats, success_rate] = classifier(categoric_var, gaze_descriptors,classifier_type,cross_validation);
        
        %         %LDA 1st Eigen vector: absolute values and normalization
        %         lda_stats.eigenvec(:,1)=abs(lda_stats.eigenvec(:,1));
        %         LDA_1st_Eigen_vect(:,istim)=lda_stats.eigenvec(:,1)/sum(lda_stats.eigenvec(:,1));
    catch
        fprintf('stimuli %u could not be classified\n',istim)
        success_rate=NaN;
        manova_stats=NaN;
    end
    
    if ~mod(istim,1)
        fprintf('stimuli %u success_rate %d\n',istim, success_rate)
    end
    
    correct_classif(istim)=success_rate;
end

fprintf('average correct classification score over %u stimuli is %3.1f %% (chance = %3.1f %%)\n',length(all_stim), 100*nanmean(correct_classif),100/length(categoric_var))
hist(correct_classif)
xlabel('Correct Classification Rate','FontSize',12,'FontWeight','bold')
ylabel('Frequency','FontSize',12,'FontWeight','bold')

% errorbar(squeeze(nanmean(abs(LDA_1st_Eigen_vect),2)),nanstd(abs(LDA_1st_Eigen_vect),0,2)/sqrt(length(all_stim)),'.')
