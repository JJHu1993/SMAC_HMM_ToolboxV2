function gaze_descriptor =extract_hmm_parameters(hmm)
%Build gaze_descriptor vector from HMM parameters: priors, transition matrix coefficients, state centres and state covariances 

%Inputs: 
%hmm = struct computed with vbhmm_learn with states sorted from left to
%right with sort_hmm_state and padded with ghost states with pad_with_ghost_states so hmm state number = max(K).
%Ouput:
%gaze_descriptor = vector containing hmm parameters (priors, transition
%matrix coefficients, state center coordinates, state covariance
%coefficients along the x and y axes).

gaze_descriptor=NaN;
if isempty(hmm)
    return
end

idx_prior=1:length(hmm.prior);
idx_transmat=(length(idx_prior)+1):(length(idx_prior)+length(hmm.trans(:)'));
idx_mean_state=[];
for i_state=1:length(hmm.pdf)
    idx_mean_state=[idx_mean_state (length(idx_prior)+length(idx_transmat)+length(idx_mean_state)+1):(length(idx_prior)+length(idx_transmat)+length(idx_mean_state)+length(hmm.pdf{i_state}.mean))];
end
idx_covar_state=[];
for i_state=1:length(hmm.pdf)
    idx_covar_state=[idx_covar_state (length(idx_prior)+length(idx_transmat)+length(idx_mean_state)+length(idx_covar_state)+1):(length(idx_prior)+length(idx_transmat)+length(idx_mean_state)+length(idx_covar_state)+length(hmm.pdf{i_state}.cov))];
end

allindex=[idx_prior idx_transmat idx_mean_state idx_covar_state];


gaze_descriptor=NaN(1,length(allindex));
gaze_descriptor(idx_prior)=hmm.prior;
gaze_descriptor(idx_transmat)=hmm.trans(:)';
mean_state=[];
for i_state=1:length(hmm.pdf)
    mean_state=[mean_state hmm.pdf{i_state}.mean];
end
gaze_descriptor(idx_mean_state)=mean_state;

covar_state=[];
for i_state=1:length(hmm.pdf)
    if size(hmm.pdf{i_state}.cov)==[1 2]
    covar_state=[covar_state hmm.pdf{i_state}.cov];
    else
        covar_state=[covar_state diag(hmm.pdf{i_state}.cov)'];
    end

end
gaze_descriptor(idx_covar_state)=covar_state;

end