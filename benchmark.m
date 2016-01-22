% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

%[counts,mean_plus,mean_minus,t] = lgcp_regression;
%[counts,mean_plus,mean_minus,t] = gp_regression;
%[counts,mean_plus,mean_minus,t] = lgcp_regression_cc;

len = 80;

% Setting out the vectors to be filled by the sampled points
sampled = zeros(len,2);

% Working out the necessary sampling frequency
inte = floor(max(t)/len);
offset = round(inte/2);

%THIS IS JUST TO AVOID ANOMILI
%offset = offset -1;

times = [inte:inte:max(t)];
times = times(1:len);

t_test = times - offset;

% Filling the sampled vector
for i = 1:len
    sampled(i,1) = direct_deaths(t_test(i));
    sampled(i,2) = indirect_deaths(t_test(i));
end

%{
figure                                % Create a new figure

subplot(2,1,1)                        % First the direct incidents
shade = fill(X,Y1,'r');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
plot(t,counts(:,1))                   % Plots the predicted mean values
hold on
plot(t_test,sampled(:,1),'o')

% Now the same but for the indirect data stream
subplot(2,1,2)
shade2 = fill(X,Y2,'r');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
hold on
plot(t,counts(:,2))
hold on
plot(t_test,sampled(:,2),'o')
%}

% This is the gap in time between two of our models predictions
gap = t(2)-t(1);

compare_mean = zeros(len,2);
            
% For both direct and indirect data streams
for h = 1:2
    % For each test time 
    for i = 1:len
        
        % Define the goal time
        goal = t_test(i);

        k = 0; j=1;
        best = [];

        while (k==0 && j<=length(t))
            % Get the next sample time
            test = t(j);
            
            % Work out the difference between the sample and model times
            dist = test-goal;

            if dist == 0
                compare_mean(i,h) = counts(j,h);
                k = 1;
                
            elseif abs(dist) < gap
                
                best = [best;test,j];
                
                if length(best(:,1)) == 2
                    frac = (goal-best(1,1))/gap;
                    
                    compare_mean(i,h) = (1-frac)*counts(best(1,2),h)+...
                        frac*counts(best(2,2),h);
                    compare_var(i,h) = (1-frac)*variances(h,best(1,2))+...
                        frac*variances(h,best(2,2));
                    
                elseif length(best(:,1)) == 1 && j==length(t)
                    compare_mean(i,h) = counts(best(1,2),h);
                    compare_var(i,h) = variances(h,best(1,2));
                end
            end
            

            j = j+1;

        end
        
        % The following two lines are only for LGCPs
        cmeanP(h,i) = exp(log(compare_mean(i,h)+2*sqrt(compare_var(i,h))));
        cmeanM(h,i) = exp(log(compare_mean(i,h)-2*sqrt(compare_var(i,h))));

    end
end

t_test = t_test(2:end-1);
sampled = sampled((2:end-1),:);
compare_mean = compare_mean((2:end-1),:);
compare_var = compare_var((2:end-1),:);

cmeanP = cmeanP(:,(2:end-1));
cmeanM = cmeanM(:,(2:end-1));


X = [t_test,fliplr(t_test)];
Y1 = [cmeanM(1,:),fliplr(cmeanP(1,:))];
Y2 = [cmeanM(2,:),fliplr(cmeanP(2,:))];

% This but will only work for LGCP but bear with me
for h = 1:2
    for i = 1:length(sampled)
        if sampled(i,h) == 0
            lsampled(i,h) = 0;
        else
            lsampled(i,h) = log(sampled(i,h));
        end
        
        if compare_mean(i,h) == 0
            compare_lmean(i,h) = 0;
        else
            compare_lmean(i,h) = log(compare_mean(i,h));
        end
    end
end

var1 = compare_var(:,1);
var2 = compare_var(:,2);
d1 = lsampled(:,1)-compare_lmean(:,1);
d2 = lsampled(:,2)-compare_lmean(:,2);

logp1 = -0.5*(len*log(2*pi)+log(ones(1,length(var1))*var1)+...
    transpose(d1)*(d1./var1));
p1 = exp(logp1)

logp2 = -0.5*(len*log(2*pi)+log(ones(1,length(var2))*var2)+...
    transpose(d2)*(d2./var2));
p2 = exp(logp2)


%figure
%plot(t_test,[sampled compare_mean])

figure
shade = fill(X,Y1,'b');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2,'g');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
plot(t_test,compare_mean)
hold on
plot(t_test,sampled,'x');


