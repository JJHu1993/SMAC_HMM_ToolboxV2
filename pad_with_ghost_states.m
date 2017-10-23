function padded_hmm=pad_with_ghost_states(hmm,K,im)
% add 'ghost states' if K < Kmax so all gaze descriptor vectors have the same dimension

%Inputs: 
%hmm = struct computed with vbhmm_learn with states sorted from left to
%right with sort_hmm_state
%K: maximum number of states used to train all hmms
%im: visual stimulus (height x width x color)
%Output: 
%padded_hmm = struct similar to hmm. If hmm state number < K, hmm is padded with 'ghost states',
%ie with priors and corresponding transition matrix coefficients = 0. 

relativeindex=1:length(hmm.prior);

if length(relativeindex)==K % if K = Kmax don't do anything
    padded_hmm=hmm;
    return
end

padded_hmm.prior=zeros(1,K)+0.0001;%Initialization priors (regularized)
padded_hmm.trans=zeros(K)+0.0001;%Initialization transition matrix

for ipdf=1:K %Initialization emissions
    padded_hmm.pdf{ipdf}.priors=1;
    padded_hmm.pdf{ipdf}.mean=[size(im,2)/2 size(im,1)/2];%center of the scene;
    padded_hmm.pdf{ipdf}.cov=[33.3 33.3];
end

if length(relativeindex)<K
    
    padded_hmm.prior(relativeindex)=hmm.prior;%sort priors
    for i=1:length(relativeindex)%sort transition matrix
        for j=1:length(relativeindex)
            padded_hmm.trans(relativeindex(i),relativeindex(j))=hmm.trans(i,j);
        end
        
        for idx=1:length(relativeindex) %sort emissions pdf
            padded_hmm.pdf{relativeindex(idx)}=hmm.pdf{idx};
        end
    end
    
end

end