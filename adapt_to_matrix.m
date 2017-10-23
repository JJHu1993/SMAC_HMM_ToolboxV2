%adapt to matrix
function out = adapt_to_matrix(data)
out = []; j = 1;
for i = 1:size(data,1)
    tout = data{i,1};
    if isempty(tout)
        continue
    end
    k = size(tout,1);
    out(j:j+k-1,:) = tout;
    j = j + k;
end