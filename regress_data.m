% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Cutting down the data to 01/07 - 12/08
direct_deaths = direct_deaths([366:1:1096]);
indirect_deaths = indirect_deaths([366:1:1096]);

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);
weeks = transpose([1:length(direct_deaths)/7]);

% Binning the data into weeks instead of days
direct_weekly = zeros(length(weeks),1);
indirect_weekly = zeros(length(weeks),1);

% Sums every 7 elements of the deaths vector to create a weekly total
for j = 1:length(weeks);
    i = 7*(j-1)+1;
    direct_weekly(j) = direct_deaths(i) + direct_deaths(i+1) + ...
        direct_deaths(i+2) + direct_deaths(i+3) + direct_deaths(i+4) ...
        + direct_deaths(i+5) + direct_deaths(i+6);
end

% Same for the indirect deaths
for j = 1:length(weeks);
    i = 7*(j-1)+1;
    indirect_weekly(j) = indirect_deaths(i) + indirect_deaths(i+1) + ...
        indirect_deaths(i+2) + indirect_deaths(i+3) + ...
        indirect_deaths(i+4) + indirect_deaths(i+5) + indirect_deaths(i+6);
end

%direct_deaths = direct_weekly/7;
%indirect_deaths = indirect_weekly/7;

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 50;

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

y = zeros(newlength,2);

% Renamed for ease of plotting
y(:,1) = direct_sampled;
y(:,2) = indirect_sampled;

%plot(t,y)

% This is my inital guess at the mean, realistically i've had to be quite
% selective, hopefully i can refine later

% It needs to be +/- 10 from the best value
step = [1 0.1 0.01;2 0.2 0.02;1 0.1 0.01];

a = [sum(y(:,1))/newlength,45,10.1;sum(y(:,2))/newlength,35,10.1];
best = 0;

% For both the indirect and direct data sets
for p = 1:2 
    % Performing decimal search, not sure whether this works
    for s = 1:3
        mew = [a(p,1)-9.5*step(1,s):step(1,s):a(p,1)+9.5*step(1,s)];
        h = [a(p,2)-9.5*step(2,s):step(2,s):a(p,2)+9.5*step(2,s)];
        lambda = [a(p,3)-9.5*step(3,s):step(3,s):a(p,3)+9.5*step(3,s)];

        for k = 1:20
            for j = 1:20
                cov = cov_matrix(t,t,h(j),lambda(k));
                for i = 1:20
                    l = likelihood(cov,y(:,p),mew(i));
                    if l >= best
                        best = l;
                        a(p,:) = [mew(i),h(j),lambda(k)];
                    end
                end
            end
        end
    end
end

% Now attempting to show how likelihood changes with variable
% The 'ideal' values for the other parameters are assumed

% Mean first

% For the mean the covariance doesn't change so working them out first
cov1 = cov_matrix(t,t,a(1,2),a(1,3));
cov2 = cov_matrix(t,t,a(2,2),a(2,3));

% The range of means you want to plot - the chosen value +/- 20
mean = [a(1)-15:0.1:a(1)+15];
% Setting up a vector for the likelihood values
mean_likelihood = zeros(length(mean),2);

% For each mean work out the direct and indirect streams likelihood
for i = 1:length(mean)
    mean_likelihood(i,1) = likelihood(cov1,y(:,1),mean(i));
    mean_likelihood(i,2) = likelihood(cov2,y(:,2),mean(i));
end

% Next h

% Setting up range and empty vector - 0 to 3 * optimum value
h = [0:0.1:a(1,2)*3];
h_likelihood = zeros(length(h),2);

% Same again, but the covariance has to be recalculated for every value of
% h - cov1 for direct, cov2 for indirect
for i = 1:length(h)
    cov1 = cov_matrix(t,t,h(i),a(1,3));
    cov2 = cov_matrix(t,t,h(i),a(2,3));
    h_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    h_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

% Lastly finding lambda
lambda = [0:0.1:a(1,3)+30];
lambda_likelihood =zeros(length(lambda),2);

% Same principal
for i = 1:length(lambda)
    cov1 = cov_matrix(t,t,a(1,2),lambda(i));
    cov2 = cov_matrix(t,t,a(1,2),lambda(i));
    lambda_likelihood(i,1) = likelihood(cov1,y(:,1),a(1,1));
    lambda_likelihood(i,2) = likelihood(cov2,y(:,2),a(2,1));
end

% Plotting the likelihood variation for all three variables
figure
subplot(2,2,1)
plot(mean,mean_likelihood)
title('Likelihood for Varying \mu')

subplot(2,2,2)
plot(h,h_likelihood)
title('Likelihood for Varying h')

subplot(2,2,3)
plot(lambda,lambda_likelihood)
title('Likelihood for Varying \lambda')

% Outputting the chosen values for the parameters
a

% Setting up a 'second' observation to compare against
direct_test = zeros(newlength,1);
indirect_test = zeros(newlength,1);

test_offset = round(inte/2);

% This is the same data but sampled 1 day later, at the same rate
for i = 1:newlength
    direct_test(i) = direct_deaths((i-1)*inte+test_offset);
    indirect_test(i) = indirect_deaths((i-1)*inte+test_offset);
end

x2 = [1:length(days)];

% The following are defined in order to reduce time complexity

cova1 = cov_matrix(x2,t,a(1,2),a(1,3));  % Direct x2 - t covariance
cova2 = cov_matrix(x2,t,a(2,2),a(2,3));  % Indirect x2 - t covariance

covb1 = cov_matrix(t,t,a(1,2),a(1,3)); % Direct t - t covariance
covb2 = cov_matrix(t,t,a(2,2),a(2,3)); % Indirect t - t covariance

% The predicted GP means for direct and indirect data streams
mean_direct = a(1,1)*ones(length(x2),1) + ...
    cova1*inv(covb1)*(y(:,1)-a(1,1)*ones(newlength,1));
mean_indirect = a(2,1)*ones(length(x2),1) + ...
    cova2*inv(covb2)*(y(:,1)-a(2,1)*ones(newlength,1));


cov_direct = cov_matrix(x2,x2,a(1,2),a(1,3)) - cova1*inv(covb1)*transpose(cova1);
cov_indirect = cov_matrix(x2,x2,a(2,2),a(2,3)) - cova2*inv(covb2)*transpose(cova2);

mean_plus_sd = ones(2,length(x2));
mean_minus_sd = ones(2,length(x2));

for i = 1:length(x2)
    mean_plus_sd(1,i) = mean_direct(i) + sqrt(cov_direct(i,i));
    mean_plus_sd(2,i) = mean_indirect(i) + sqrt(cov_indirect(i,i));
    mean_minus_sd(1,i) = mean_direct(i) - sqrt(cov_direct(i,i));
    mean_minus_sd(2,i) = mean_indirect(i) - sqrt(cov_indirect(i,i));
end

X = [x2,fliplr(x2)];
Y1 = [mean_minus_sd(1,:),fliplr(mean_plus_sd(1,:))];
Y2 = [mean_minus_sd(2,:),fliplr(mean_plus_sd(2,:))];

figure 
subplot(2,1,1)
shade = fill(X,Y1,'r');
set(shade,'facealpha',.2)
hold on
plot(x2,[mean_direct])
hold on
plot((t+(test_offset-inte)),[direct_test],'o')
title('Direct Fire Incidents')
xlabel('Time /Days')
ylabel('Incidents')


subplot(2,1,2)
shade = fill(X,Y2,'r');
set(shade,'facealpha',.2)
hold on
plot(x2,[mean_indirect])
hold on
plot((t+(test_offset-inte)),[indirect_test],'o')
title('Indirect Fire Incidents')
xlabel('Time /Days')
ylabel('Incidents')
