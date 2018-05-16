% File: BanditTrainingAlgorithmUser.m
% Authors: Matthew Stock & Richard Kneale
% Date created: 14th February 2017
% Description: Displays the results of teaching an agent an optimal path 
% using three values of epsilon

% The number of tests to conduct
numberOfTests = 2000;

% Used to achieve a smoothing effect
episodesToAverageOver = 100;
% The number of plays over which the agent will learn
numberOfPlays = 1000;

% Number of arms belonging to the test bed
numberOfArms = 10;

% The input required to retrieve random values from a Gaussian distribution
gaussianDistributionMean = 0;
gaussianDistributionStandardDeviation = 1;

% The values of epsilon to be used in training the agents
epsilon1 = 0;
epsilon2 = 0.01;
epsilon3 = 0.1;

% Set up the plays matrix for the x-axis
plays = (0 : numberOfPlays);

% A matrix to represent the potential rewards for pulling each arm
% Each column represents an arm
% The reward values are generated using a Gaussian distribution
meanRewardForArms = normrnd(gaussianDistributionMean, gaussianDistributionStandardDeviation, ...
                             1, numberOfArms);

% Return the averageRewardMatrix1 and percentageOfOptimalAction1 using epsilon1
[averageReward1, percentageOfOptimalAction1] = BanditTrainingAlgorithm(meanRewardForArms, ...
                            epsilon1, numberOfTests, numberOfPlays, episodesToAverageOver);

% Return the averageRewardMatrix2 and percentageOfOptimalAction2 when epsilon2
[averageReward2, percentageOfOptimalAction2] = BanditTrainingAlgorithm(meanRewardForArms, ...
                            epsilon2, numberOfTests, numberOfPlays, episodesToAverageOver);

% Return the averageRewardMatrix3 and percentageOfOptimalAction3 when epsilon3
[averageReward3, percentageOfOptimalAction3] = BanditTrainingAlgorithm(meanRewardForArms, ...
                            epsilon3, numberOfTests, numberOfPlays, episodesToAverageOver);

% Output the figure
% Average performance of epsilon-greedy action-value methods on the n-armed
% testbed. These data are averages over t tasks. All methods used
% sampled averages as their action-value estimates.
figure;

% Figure representing the average reward as a function of plays
subplot(211);
plot(plays, averageReward1, plays, averageReward2, plays, averageReward3);
xlabel('Plays');
ylabel('Average reward');
legend(['epsilon = ' num2str(epsilon1)], ['epsilon = ' num2str(epsilon2)], ...
       ['epsilon = ' num2str(epsilon3)], 'Location', 'southeast');

% Figure representing the percentage of optimal action as a function of plays
subplot(212);
plot(plays, percentageOfOptimalAction1, plays, percentageOfOptimalAction2, ...
     plays, percentageOfOptimalAction3);
xlabel('Plays');
ylabel('% Optimal action');
legend(['epsilon = ' num2str(epsilon1)], ['epsilon = ' num2str(epsilon2)], ...
       ['epsilon = ' num2str(epsilon3)], 'Location', 'southeast');