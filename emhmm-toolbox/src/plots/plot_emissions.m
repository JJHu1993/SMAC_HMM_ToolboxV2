function plot_emissions(data, gamma, pdf, img)
% plot_emissions - plot emission densities
%
%  Plots ellipses for each ROI. The ellipse is 2 standard-deviations (95% percentile).
%
%     plot_emissions(data, gamma, pdf, img)
%
%  INPUTS
%    data  = data (same format as used by vbhmm_learn)
%    gamma = assignment variables for each data point to emission density
%    pdf   = the emission densities (from vbhmm)
%    img   = image for background
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong


% VERSIONS
% 2017-01-17: ABC - add duration as marker size.

color = get_color_list();
if ~isempty(img)
  imshow(img);
end
axis ij
hold on

if ~isempty(data)
  out = cat(1, data{:});
  gamma = cat(2, gamma{:});
  [~,cl] = max(gamma,[],1);
  K = size(gamma,1);
  
  D = size(data{1}, 2);
  if D > 2
    myscale = get_duration_rescale(data);
  end

  for k=1:K
    ii = find(cl==k);
    if (D == 2)
      scatter(out(ii,1),out(ii,2), 6, [color(k) 'o']);
    else      
      scatter(out(ii,1),out(ii,2), myscale(out(ii,3)),  ...
        [color(k) 'o']);
    end
  end
else
  K = length(pdf);
end

for k = 1:K
  mu = pdf{k}.mean;
  sigma(:,:) = pdf{k}.cov;
  plot2D(mu,sigma,color(k))
  %text(mu(1), mu(2), sprintf('%d', k));
end
for k = 1:K
  mu = pdf{k}.mean;
  text(mu(1), mu(2), sprintf('%d', k), 'color', color(k), ...
    'horizontalalignment', 'center');
end
hold off
title('ROIs');

%% plot a Gaussian as ellipse
function plot2D(mu,Sigma,color)
% truncate to 2D
mu = mu(1:2);
Sigma = Sigma(1:2,1:2);

mu = mu(:);
if ~any(isnan(Sigma(:))) && ~any(isinf(Sigma(:)))
  [U,D] = eig(Sigma);
  n = 100;
  t = linspace(0,2*pi,n);
  xy = [cos(t);sin(t)];
  k = sqrt(conf2mahal(0.95,2));
  w = (k*U*sqrt(D))*xy;
  z = repmat(mu,[1 n])+w;
  h = plot(z(1,:),z(2,:),'Color',color,'LineWidth',1);
end

function m = conf2mahal(c,d)
m = chi2inv(c,d);