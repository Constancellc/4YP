function regress_data

% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Cutting down the data to 01/07 - 12/08
direct_deaths = direct_deaths([366:1:1096]);
indirect_deaths = indirect_deaths([366:1:1096]);

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 25;

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
h = [1:5];
lambda = [20:40:100];
noise = [0];
%h =2;
%lambda = 100;
%noise = 0;

% This is to suppress the output of the optimizer. It's irritating.
options = optimset('Display', 'off');

% This cell array contains function handles for fmincon
test = {@test_d,@test_i};

% For both the direct and indirect data streams
for i = 1:2
    
    % Starting with a (hopefully) comparitively small likelihood.
    best = -10^6;
    
    for j = 1:length(h)        
        for k = 1:length(lambda)
            for m = 1:length(noise)
                
                % Find the local minima of 1/liklelihood function
                val = fmincon(test(i),[mean(i),h(j),lambda(k),noise(m)]...
                    ,[],[],[],[],[0,0,0,0],[30,50,700,0.1],[],options);
                
                % Work out the resulting likelihood for the chosen
                % hyperparameters
                if i == 1
                    l = 1/test_d(val);
                elseif i == 2
                    l = 1/test_i(val);
                end
                
                % If the likelihood is higher than the previous 'best
                % guess' replace the currently accepted parameters
                if l >= best
                    best = l;
                    a(i,:) = val;
                end
            end
        end
    end
end

% This is a vector of all of the days in the observed period, for plotting
x2 = [1:length(days)];

% The following are defined in order to reduce time complexity

cova1 = cov_matrix(x2,t,a(1,2),a(1,3),a(1,4));  % Direct x2 - t covariance
cova2 = cov_matrix(x2,t,a(2,2),a(2,3),a(2,4));  % Indirect " "

covb1 = cov_matrix(t,t,a(1,2),a(1,3),a(1,4)); % Direct t - t covariance
covb2 = cov_matrix(t,t,a(2,2),a(2,3),a(2,4)); % Indirect " "

% Because this would otherwise be worked out twice
icovb1 = covb1\eye(size(covb1));
icovb2 = covb1\eye(size(covb1));

% The predicted GP means for direct and indirect data streams
mean_direct = a(1,1)*ones(length(x2),1) + ...
    cova1*icovb1*(y(:,1)-a(1,1)*ones(newlength,1));
mean_indirect = a(2,1)*ones(length(x2),1) + ...
    cova2*icovb2*(y(:,2)-a(2,1)*ones(newlength,1));


%cov_direct = cov_matrix(x2,x2,a(1,2),a(1,3),a(1,4)) - cova1*inv(covb1)*transpose(cova1);
%cov_indirect = cov_matrix(x2,x2,a(2,2),a(2,3),a(2,4)) - cova1*inv(covb2)*transpose(cova2);

% Note the h-squared term + noise is only true for s.e kernel function - 
% should be the leading diagonal element of whatever covariance function we
% go for.

cov_direct = (a(1,2)^2+a(1,4))*ones(length(x2),1) - ...
    diagonly3(cova1,icovb1,transpose(cova1));
cov_indirect = (a(2,2)^2+a(2,4))*ones(length(x2),1) - ...
    diagonly3(cova2,icovb2,transpose(cova2));

% Sets up a vector to contain the error bound terms for plotting
mean_plus_sd = ones(2,length(x2));
mean_minus_sd = ones(2,length(x2));

% Fills in the +/- 1 std vectors for both streams
for i = 1:length(x2)
    mean_plus_sd(1,i) = mean_direct(i) + 2*sqrt(abs(cov_direct(i)));
    mean_plus_sd(2,i) = mean_indirect(i) + 2*sqrt(abs(cov_indirect(i)));
    mean_minus_sd(1,i) = mean_direct(i) - 2*sqrt(abs(cov_direct(i)));
    mean_minus_sd(2,i) = mean_indirect(i) - 2*sqrt(abs(cov_indirect(i)));
end

% Setting up a 'second' observation to compare against
direct_test = zeros(newlength,1);
indirect_test = zeros(newlength,1);

test_offset = round(inte/2);

% This is the same data but sampled 1 day later, at the same rate
for i = 1:newlength
    direct_test(i) = direct_deaths((i-1)*inte+test_offset);
    indirect_test(i) = indirect_deaths((i-1)*inte+test_offset);
end


% These are the vectors to be used in the shade effect
X = [x2,fliplr(x2)];
Y1 = [mean_minus_sd(1,:),fliplr(mean_plus_sd(1,:))];
Y2 = [mean_minus_sd(2,:),fliplr(mean_plus_sd(2,:))];

figure 
subplot(2,1,1)                        % Direct Fire first
shade = fill(X,Y1,'r');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
hold on
plot(x2,[mean_direct])                % Plots the predicted mean
hold on
plot((t+(test_offset-inte)),[direct_test],'o')
title('Direct Fire Incidents')
xlabel('Time /Days')
ylabel('Incidents')

subplot(2,1,2)                        % Now same but for indirect fire
shade2 = fill(X,Y2,'r');
set(shade2,'facealpha',.2)
hold on
plot(x2,[mean_indirect])
hold on
plot((t+(test_offset-inte)),[direct_test],'o')
title('Indirect Fire Incidents')
xlabel('Time /Days')
ylabel('Incidents')
           
a

%--------------------------------------
% START OF LIKELIHOOD PLOTTING SECTION
%--------------------------------------
%{

% For the mean the covariance doesn't change so working them out first
cov1 = cov_matrix(t,t,a(1,2),a(1,3),a(1,4));
cov2 = cov_matrix(t,t,a(2,2),a(2,3),a(2,4));

% Chosing ranges to plot
mean = [a(1,1)-30:0.1:a(1,1)+30];
h = [1:0.1:50];
lambda = [1:0.1:300];
noise = [0:0.01:30];

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
    cov1 = cov_matrix(t,t,h(i),a(1,3),a(1,4));
    cov2 = cov_matrix(t,t,h(i),a(2,3),a(2,4));
    h_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    h_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

for i = 1:length(lambda)
    cov1 = cov_matrix(t,t,a(1,2),lambda(i),a(1,4));
    cov2 = cov_matrix(t,t,a(1,2),lambda(i),a(2,4));
    lambda_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    lambda_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

for i = 1:length(noise)
    cov1 = cov_matrix(t,t,a(1,2),a(1,3),noise(i));
    cov2 = cov_matrix(t,t,a(1,2),a(2,3),noise(i));
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
    function l = test_d(x)
        % Input x - a vector [mean, h, lambda, noise variance]
        
        cov = cov_matrix(t,t,x(2),x(3),x(4));
        
        mag = det(cov);         % Determinant of covariance
        p = length(cov);        % Number of input variables
        
        % A vector of the difference between the output values and the
        % given mean
        d = y(:,1)-x(1)*ones(p,1);

        % Second two terms of log-likelihood
        l = -(1/2)*log(mag)-(1/2)*transpose(d)*(cov\eye(p,p))*d;
        l = 1/l;                % Because our optimiser minmizes 
    end

% Predictably this one tests the indirect fire stream
    function l = test_i(x)
        % Input x - a vector [mean, h, lambda, noise variance]
        
        cov = cov_matrix(t,t,x(2),x(3),x(4));
        
        mag = det(cov);         % Determinant of covariance
        p = length(cov);        % Number of input variables
        
        % A vector of the difference between the output values and the
        % given mean
        d = y(:,2)-x(1)*ones(p,1);

        % Second two terms of log-likelihood
        l = -(1/2)*log(mag)-(1/2)*transpose(d)*(cov\eye(p,p))*d;
        l = 1/l;                % Because our optimiser minmizes 
    end
end
