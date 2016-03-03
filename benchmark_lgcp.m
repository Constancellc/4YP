% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

len = 500;
inte = floor(max(t)/len);
offset = round(inte/2);

% This tries to make it so that we are only interpolating, as it's hard to
% extrapolate from LGCP models
times = [t(1):inte:length(direct_deaths)];
t_test = times(1:len);

% This insures that we don't get too much overlap between the training and
% test data
t_test = t_test + offset;               

sampled = zeros(len,2);
lsampled = zeros(len,2);

% Filling the sampled vector
for i = 1:len
    sampled(i,1) = direct_deaths(t_test(i));
    
    if sampled(i,1) == 0
        lsampled(i,1) = 0;
    else
        lsampled(i,1) = log(sampled(i,1));
    end
    
    sampled(i,2) = indirect_deaths(t_test(i));
    
    if sampled(i,2) == 0
        lsampled(i,2) = 0;
    else
        lsampled(i,2) = log(sampled(i,2));
    end
end

% This is the gap in time between two of our models predictions
gap = t(2)-t(1);

% This vectors will contain the comparison data
compare_lmean = zeros(len,2);
compare_var = zeros(len,2);

cmeanP = zeros(2,len);
cmeanM = zeros(2,len);

% We want to find the predictions for the inputs closest to the inputs of
% the test data
            
% For both direct and indirect data streams
for h = 1:2
    % For each test time 
    for i = 1:len
        
        % Define the goal time
        goal = t_test(i);

        k = 0; j=1;
        best = [];

        % While we don't have an answer and the counter is less than the
        % number of predictions.
        while (k==0 && j<=length(t))
            % Get the next sample time
            test = t(j);
            
            % Work out the difference between the sample and model times
            dist = test-goal;

            if dist == 0
                compare_lmean(i,h) = log(counts(j,h));
                compare_var(i,h) = variances(j,h);
                k = 1;
                
            elseif abs(dist) < gap
                % This will be true for the two closest values
                
                best = [best;test,j];
                
                if length(best(:,1)) == 2
                    % At this point best is a 2x2 matrix containing the two
                    % closest times and the corresponding indicies
                    
                    frac = (goal-best(1,1))/gap;
                    
                    compare_lmean(i,h) = log((1-frac)*counts(best(1,2),h)+...
                        frac*counts(best(2,2),h));
                    compare_var(i,h) = (1-frac)*variances(best(1,2),h)+...
                        frac*variances(best(2,2),h);
                    
                elseif length(best(:,1)) == 1 && j==length(t)
                    compare_lmean(i,h) = log(counts(best(1,2),h));
                    compare_var(i,h) = variances(best(1,2),h);
                end
            end
            

            j = j+1;

        end
        
        cmeanP(h,i) = exp(compare_lmean(i,h)+2*sqrt(compare_var(i,h)));
        cmeanM(h,i) = exp(compare_lmean(i,h)-2*sqrt(compare_var(i,h)));

    end
end

X = [t_test,fliplr(t_test)];
Y1 = [cmeanM(1,:),fliplr(cmeanP(1,:))];
Y2 = [cmeanM(2,:),fliplr(cmeanP(2,:))];

errors = exp(compare_lmean) - sampled;
square_errors = errors.^2;

% This is the first test measure - the mean square error
mse1 = (1/len)*sum(square_errors(:,1))
mse2 = (1/len)*sum(square_errors(:,2))

d = compare_lmean - lsampled;

d1 = d(:,1);
d2 = d(:,2);
var1 = compare_var(:,1);
var2 = compare_var(:,2);


p1 = 0;
p2 = 0;
for i = 1:len
    p1 = p1 -0.5*log(2*pi*var1(i)) -(d1(i))^2/(2*var1(i));
    p2 = p2 -0.5*log(2*pi*var2(i)) -(d2(i))^2/(2*var2(i));
end

p1 = exp(p1/len)
p2 = exp(p2/len)


figure
shade = fill(X,Y1,'b');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2,'g');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
plot(t_test,exp(compare_lmean))
hold on
plot(t_test,sampled,'x');


