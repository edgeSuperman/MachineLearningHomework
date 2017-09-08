  X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
  Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
  X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];
  Y2 = Y1.^0.5 + Y1;

  X = (X2(:,2:4));
  
  X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
  
mu = mean(X);
sigma = std(X);
X_norm = (X - mu); %  * sigma';

c = length(sigma);

for i = 1:1:c
    s = sigma(:,i);
    s = 1/s;
   X_norm(:,i) = s * X_norm(:,i); 
end

