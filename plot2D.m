function plot2D(mu,Sigma,color)
%color = 'r';
mu = mu(:);
try
    [U,D] = eig(Sigma);
catch
    [U,D] = eig(diag(Sigma));
end
n = 100;
t = linspace(0,2*pi,n);
xy = [cos(t);sin(t)];
k = sqrt(conf2mahal(0.95,2));
w = (k*U*sqrt(D))*xy;
z = repmat(mu,[1 n])+w;
h = plot(z(1,:),z(2,:),'Color',color,'LineWidth',2);
end

function m = conf2mahal(c,d)
m = chi2inv(c,d);
end