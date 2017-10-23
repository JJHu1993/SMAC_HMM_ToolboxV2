function  sorted_hmm = sort_hmm_state(hmm)
%Sort states from left to right
%Input: hmm struct computed from vbhmm_learn
%Output: same hmm struct, with states sorted from left to right
sorted_hmm=hmm;
centre=[];

for iclus=1:length(hmm.prior)
    centre=[centre; hmm.pdf{iclus}.mean];
end

[~, index]=sortrows(centre,1); 
sorted_hmm.prior=hmm.prior(index);
sorted_hmm.N=hmm.N(index);
sorted_hmm.N1=hmm.N1(index);

for i=1:length(index)
    for j=1:length(index)
        sorted_hmm.trans(i,j)=hmm.trans(index(i),index(j));
        sorted_hmm.M(i,j)=hmm.M(index(i),index(j));
    end
    
    for idx=1:length(index)
        sorted_hmm.pdf{idx}=hmm.pdf{index(idx)};
    end
    
    for iscan=1:length(hmm.gamma)
        sorted_hmm.gamma{iscan}=hmm.gamma{iscan}(index,:);
    end
end