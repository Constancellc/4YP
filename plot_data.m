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

% Compares the daily direct and indirect deaths
plot(days,[direct_deaths indirect_deaths])

% Compares the weekly direct and indirect deaths
plot(weeks,[direct_weekly indirect_weekly])
