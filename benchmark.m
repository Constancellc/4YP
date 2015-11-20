% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

[counts,mean_plus,mean_minus,t] = lgcp_regression;

X = [t,fliplr(t)];
Y1 = [mean_minus(1,:),fliplr(mean_plus(1,:))];
Y2 = [mean_minus(2,:),fliplr(mean_plus(2,:))];

%len = length(t);
len = 50;

% Setting out the vectors to be filled by the sampled points
sampled = zeros(len,2);

% Working out the necessary sampling frequency
inte = floor(length(direct_deaths)/len);
offset = round(inte/2);

%THIS IS JUST TO AVOID ANOMILI
%offset = offset -1;

times = [inte:inte:length(direct_deaths)];
times = times(1:len);

t_test = times - offset;

% Filling the sampled vector
for i = 1:len
    sampled(i,1) = direct_deaths(t_test(i));
    sampled(i,2) = indirect_deaths(t_test(i));
end

% {
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

gap = t(2)-t(1);

compare = zeros(len,2);
            
% For both direct and indirect data streams
for h = 1:2
    % For each test time 
    for i = 1:len
        % Define the goal time
        goal = t_test(i);

        k = 0; j=1;
        best = 100000;

        while (k==0 && j<=length(t))
            % Get the next sample time
            test = t(j);
            % Work out the difference between the sample and model times
            dist = test-goal;

            if dist == 0
                compare(i,h) = counts(j,h);
                k = 1;
                
            elseif abs(dist) < gap

                if abs(dist) < best
                    best = abs(test-goal);
                    frac = abs(dist/gap);

                    if j==length(t)
                        compare(i,h) = frac*counts(j,h) + (1-frac)*counts(j-1,h);
                    elseif (dist<0 || j==1)
                        compare(i,h) = frac*counts(j,h) + (1-frac)*counts(j+1,h);
                    else
                        compare(i,h) = frac*counts(j,h) + (1-frac)*counts(j-1,h);
                    end
                end
            end

            j = j+1;

        end

    end
end 

errors = compare-sampled;
figure
plot([1:len],errors)

direct_error = sqrt(sum(errors(:,1).^2)/len)
indirect_error = sqrt(sum(errors(:,2).^2)/len)
