function [counts,mean_plus,mean_minus,x2,a] = gp_regression

% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Cutting down the data to 01/07 - 12/08
%direct_deaths = direct_deaths([366:1:1096]);
%indirect_deaths = indirect_deaths([366:1:1096]);

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 100;

% Setting out the vectors to be filled by the sampled points
direct_sampled = zeros(newlength,1);
indirect_sampled = zeros(newlength,1);

% Working out the necessary sampling frequency
inte = floor(length(direct_deaths)/newlength);

% Filling the sampled vector
for i = 1:newlength
    direct_sampled(i) = direct_deaths(i*inte);
    indirect_sampled(i) = indirect_deaths(i*inte);
end
    
% A vector of the number of days into conflict for plot
t = [inte:inte:length(direct_deaths)];
if length(t) >= newlength + 1
    t = t(1:newlength);
end

%y = zeros(newlength,2);
y = [direct_sampled,indirect_sampled];

% This is the matrix which is going to contain the optimized hyperparamters
a = zeros(2,4);

% The function fmincon computes a local minimum so i'm using a 'grid
% search' to start the optimizer at multilple points, then select the best
% one. The following vectors contain the start points.

% Picking the starting points
mean = [sum(y(:,1))/length(t),sum(y(:,2))/length(t)];
%h = [1 10];                  % To save time, i've just started with what i
%lambda = [50 100];           % know is a good guess (below)
%noise = [0 4 20];
h = 5;
lambda = 100;
noise = 10;

% This is to suppress the output of the optimizer. It's irritating.
%options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');
options = optimset('Display', 'off');

% This cell array contains function handles for fmincon
%test = {@test_d,@test_i};

% For both the direct and indirect data streams
for i = 1:2
    
    % Starting with a (hopefully) comparitively large number.
    best = 10^6;
%{
    for j = 1:length(h)        
        for k = 1:length(lambda)
            for m = 1:length(noise)
%}
    
    [x,fval] = fmincon(@test,[mean(i),5,20,10]...
                    ,[],[],[],[],[0,0,0,0],[30,50,700,100],[],options);
    
    if fval <= best
        best = fval;
        a(i,:) = x;
    end
    
end

% This is a vector of all of the days in the observed period, for plotting
x2 = [1:max(days)];

% The following are defined in order to reduce time complexity

cova1 = cov_matrix(x2,t,a(1,2),a(1,3));  % Direct x2 - t covariance
cova2 = cov_matrix(x2,t,a(2,2),a(2,3));  % Indirect " "

len = length(t); 

covb1 = cov_matrix(t,t,a(1,2),a(1,3)) + a(1,4)*eye(len,len);
covb2 = cov_matrix(t,t,a(2,2),a(2,3)) + a(2,4)*eye(len,len); 

icovb1 = covb1\eye(size(covb1));
icovb2 = covb2\eye(size(covb2));


% The predicted GP means for direct and indirect data streams
mean_direct = a(1,1)*ones(length(x2),1) + ...
    cova1*(covb1\(y(:,1)-a(1,1)*ones(newlength,1)));
mean_indirect = a(2,1)*ones(length(x2),1) + ...
    cova2*(covb2\(y(:,2)-a(2,1)*ones(newlength,1)));

counts = [mean_direct,mean_indirect];

% Note the h-squared term + noise is only true for s.e kernel function - 
% should be the leading diagonal element of whatever covariance function we
% go for.

cov_direct = (a(1,2)^2+a(1,4))*ones(length(x2),1) - ...
    diagonly3(cova1,icovb1,transpose(cova1));
cov_indirect = (a(2,2)^2+a(2,4))*ones(length(x2),1) - ...
    diagonly3(cova2,icovb2,transpose(cova2));

% Sets up a vector to contain the error bound terms for plotting
mean_plus = ones(2,length(x2));
mean_minus = ones(2,length(x2));

% Fills in the +/- 1 std vectors for both streams
for i = 1:length(x2)
    mean_plus(1,i) = mean_direct(i) + 2*sqrt(abs(cov_direct(i)));
    mean_plus(2,i) = mean_indirect(i) + 2*sqrt(abs(cov_indirect(i)));
    mean_minus(1,i) = mean_direct(i) - 2*sqrt(abs(cov_direct(i)));
    mean_minus(2,i) = mean_indirect(i) - 2*sqrt(abs(cov_indirect(i)));
end


% These are the vectors to be used in the shade effect
X = [x2,fliplr(x2)];
Y1 = [mean_minus(1,:),fliplr(mean_plus(1,:))];
Y2 = [mean_minus(2,:),fliplr(mean_plus(2,:))];
% {
figure 
subplot(2,2,1)                        % Direct Fire first
shade = fill(X,Y1,'g');               % Fills the confidence region   
set(shade,'facealpha',.3)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
plot(x2,[mean_direct])                % Plots the predicted mean
hold on                               % Adding a 'test' data series
%plot(t_test,[direct_test],'or')
%hold on
plot(t,y(:,1),'x')                    % Adding x's for observed points
title('Direct Fire Incidents - Training')
xlabel('Time /Days')
ylabel('Incidents')

indirect_deaths(1328) = 0;
direct_deaths(1328) = 0;

subplot(2,2,2)                        % Now same but for indirect fire
shade2 = fill(X,Y2,'g');
set(shade2,'facealpha',.3)
set(shade2,'EdgeColor','None')
hold on
plot(x2,[mean_indirect])
%hold on
%plot(x2,indirect_deaths)
hold on
plot(t,y(:,2),'x')
title('Indirect Fire Incidents - Training')
xlabel('Time /Days')
ylabel('Incidents')

subplot(2,2,3)
shade2 = fill(X,Y2,'g');
set(shade2,'facealpha',.3)
set(shade2,'EdgeColor','None')
hold on
plot(x2,[mean_indirect])
hold on
plot(x2,direct_deaths)
title('Direct Fire Incidents - Testing')
xlabel('Time /Days')
ylabel('Incidents')

subplot(2,2,4)
shade2 = fill(X,Y2,'g');
set(shade2,'facealpha',.3)
set(shade2,'EdgeColor','None')
hold on
plot(x2,[mean_indirect])
hold on
plot(x2,indirect_deaths)
title('Indirect Fire Incidents - Testing')
xlabel('Time /Days')
ylabel('Incidents')
           


%--------------------------------------
% START OF LIKELIHOOD PLOTTING SECTION
%--------------------------------------
%{

% For the mean the covariance doesn't change so working them out first
cov1 = cov_matrix(t,t,a(1,2),a(1,3)) + a(1,4)*eye(len,len);
cov2 = cov_matrix(t,t,a(2,2),a(2,3)) + a(2,4)*eye(len,len);

% Chosing ranges to plot
mean = [a(1,1)-30:0.1:a(1,1)+30];
h = [1:0.1:30];
lambda = [1:0.1:300];
noise = [0.1:0.1:300];

% Setting up vector to contain the results
mean_likelihood = zeros(length(mean),2);
h_likelihood = zeros(length(h),2);
lambda_likelihood =zeros(length(lambda),2);
noise_likelihood =zeros(length(noise),2);

% Filling vectors with results
for i = 1:length(mean)
    mean_likelihood(i,1) = likelihood(cov1,y(:,1),mean(i));
    mean_likelihood(i,2) = likelihood(cov2,y(:,2),mean(i));
end

for i = 1:length(h)
    cov1 = cov_matrix(t,t,h(i),a(1,3)) + a(1,4)*eye(len,len);
    cov2 = cov_matrix(t,t,h(i),a(2,3)) + a(2,4)*eye(len,len);
    h_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    h_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

for i = 1:length(lambda)
    cov1 = cov_matrix(t,t,a(1,2),lambda(i)) + a(1,4)*eye(len,len);
    cov2 = cov_matrix(t,t,a(2,2),lambda(i)) + a(2,4)*eye(len,len);
    lambda_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    lambda_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

for i = 1:length(noise)
    cov1 = cov_matrix(t,t,a(1,2),a(1,3)) + noise(i)*eye(len,len);
    cov2 = cov_matrix(t,t,a(2,2),a(2,3)) + noise(i)*eye(len,len);
    noise_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    noise_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

% Plotting the likelihood variation for all three variables
figure
subplot(2,2,1)
semilogy(mean,mean_likelihood)
title('Likelihood for Varying \mu')

subplot(2,2,2)
semilogy(h,h_likelihood)
title('Likelihood for Varying h')

subplot(2,2,3)
semilogy(lambda,lambda_likelihood)
title('Likelihood for Varying \lambda')

subplot(2,2,4)
semilogy(noise,noise_likelihood)
title('Likelihood for Varying \sigma ^2')

%}
%------------------------------------
% END OF LIKELIHOOD PLOTTING SECTION
%------------------------------------


%----------------------------
% LOCAL FUNCTIONS DEFINITION
%----------------------------

% These are the functions written to be used in the minimisation. They
% compute only the last two terms of the log-likelihood (as the first 
% will be the same if you aren't changing the input x-values)

% The first one tests the direct fire stream
    function [f] = test(x)
        % Input x - a vector [mean, h, lambda, noise variance]
        
        x_m = x(1);
        x_h = x(2);
        x_l = x(3);
        x_s = x(4);
                
        cov = cov_matrix(t,t,x_h,x_l);
        
        cov = cov + x_s*eye(size(cov));
        
        mag = det(cov);         % Determinant of covariance
        p = length(cov);        % Number of input variables
        
        % A vector of the difference between the output values and the
        % given mean
        d = y(:,i)-x_m*ones(p,1);

        % Second two terms of log-likelihood
        f = (1/2)*log(mag)+(1/2)*transpose(d)*(cov\d);
        

    end

end
