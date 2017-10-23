function plot_hmm_state(hmm,data,imageframe)

color = ['r','g','b','y', 'm', 'k'];
imshow(imageframe,[])
hold on
out = adapt_to_matrix(data);
scatter(out(:,1),out(:,2),'w');
for istate = 1:length(hmm.prior)
    mu = hmm.pdf{istate}.mean;
    try
    sigma(:,:) =hmm.pdf{istate}.cov;
    catch
    sigma(:,:) =diag(hmm.pdf{istate}.cov);
    end
    plot2D(mu,sigma,color(istate))
end

hold off
end