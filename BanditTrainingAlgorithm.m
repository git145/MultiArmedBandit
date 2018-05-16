% File: BanditTrainingAlgorithm.m
% Authors: Matthew Stock & Richard Kneale
% Date created: 14th February 2017
% Description: Teaches an agent an optimal path over a number of episodes

function[averageRewardMatrix, percentageOfOptimalActionMatrix] = BanditTrainingAlgorithm(meanRewardForArms, ...
                                            epsilon, numberOfTests, numberOfEpisodes, episodesToAverageOver)
    
    % Create an empty matrix to represent the rolling average reward
    % The first element must maintain a value of zero to represent the origin
    temporaryAverageRewardMatrix = zeros(numberOfTests, (numberOfEpisodes + 1));
    
    % Create an empty matrix to represent the percentage of optimal action
    % This is the percentage of times the bandit with the highest mean is chosen
    % The first element must maintain a value of zero to represent the origin
    temporaryPercentageOfOptimalActionMatrix = zeros(numberOfTests, (numberOfEpisodes + 1));
    
    % Find the index of the arm that has the highest mean reward
    [~, optimalRewardArm] = max(meanRewardForArms); 
    
    % The number of arms on the test bed is equal to the number of columns in the
    % meanRewardForArms matrix
    numberOfArms = size(meanRewardForArms, 2);
    
    % Used to calulate whether an exploitative or exploratory move should
    % be taken
    if(epsilon > 0)
        exploitativeOrExploratoryDivisor = (1 / epsilon);
    
    % Epsilon is less than or equal to zero
    else
        
        % This prevents a divide by zero error
        exploitativeOrExploratoryDivisor = (numberOfEpisodes + 1);
    end

    % Conduct the tests
    for testNumber = (1 : numberOfTests)
        % A matrix to represent the average reward obtained from using each arm
        % Each column represents an arm
        % Should be reset after each test
        averageRewardPerArm = zeros(1, numberOfArms);

        % A matrix to represent the number of times each arm has been pulled
        % Each column represents an arm
        % Should be reset after each test
        armPullCounter = zeros(1, numberOfArms);

        % Each episode
        for episodeNumber = (1 : numberOfEpisodes)
            
            % It is the first episode or an exploratory episode
            if((episodeNumber == 1) || ((mod(episodeNumber, exploitativeOrExploratoryDivisor) == 0) && ...
                    (epsilon > 0) && (episodeNumber >= exploitativeOrExploratoryDivisor)))
                % Select a random arm to pull
                armToPull = randi([1 numberOfArms]);
            
            % It is not the first episode or an exploratory episode
            else
                % armToPull is equal to the index of the arm that has so far
                % provided the highest average reward
                [~, armToPull] = max(averageRewardPerArm); 
            end
            
            % Used to calculate the percentageOfOptimalAction
            if armToPull == optimalRewardArm
                temporaryPercentageOfOptimalActionMatrix(testNumber, (episodeNumber + 1)) = 1;
            end

            % Obtain the value for the reward
            reward = normrnd(meanRewardForArms(1, armToPull), 1);

            % Update armPull
            armPullCounter(1, armToPull) = (armPullCounter(1, armToPull) + 1);
            averageRewardPerArm(1, armToPull) = (((averageRewardPerArm(1, armToPull) * ...
                                                  (armPullCounter(1, armToPull) - 1)) + ... 
                                                   reward) / ... 
                                                  (armPullCounter(1, armToPull)));

            % Update the averageRewardMatrix
            % It is the first episode
            if (episodeNumber == 1)
                temporaryAverageRewardMatrix(testNumber, (episodeNumber + 1)) = reward;
            
            % It is not the first episode
            else
                temporaryAverageRewardMatrix(testNumber, (episodeNumber + 1)) = ...
                                (((temporaryAverageRewardMatrix(1, episodeNumber) * ...
                                  (episodeNumber - 1)) + reward) / ...
                                  (episodeNumber));
            end
        end
    end
    
    % Create the smoothed reward matrix
    % Calculate the average reward and percentage of optimal action over the tests
    if(numberOfTests > 1)
        temporaryAverageRewardMatrix = (sum(temporaryAverageRewardMatrix) / numberOfTests);
        temporaryPercentageOfOptimalActionMatrix = ((sum(temporaryPercentageOfOptimalActionMatrix) / numberOfTests) * 100);
    end
    
    averageRewardMatrix = zeros(1, (numberOfEpisodes + 1));
    percentageOfOptimalActionMatrix = zeros(1, (numberOfEpisodes + 1));
    
    % Each episode
    for episodeNumber = (1 : numberOfEpisodes)

        % Write the cumulativeReward to the rewardPerEpisodeMatrix
        if(episodeNumber >= episodesToAverageOver)
            averageReward = (sum(temporaryAverageRewardMatrix(1, ((episodeNumber - episodesToAverageOver + 2) : (episodeNumber + 1)))) / episodesToAverageOver);
            averageRewardMatrix(1, (episodeNumber + 1)) = averageReward;
            
            averagePercentageOfOptimalAction = (sum(temporaryPercentageOfOptimalActionMatrix(1, ((episodeNumber - episodesToAverageOver + 2) : (episodeNumber + 1)))) / episodesToAverageOver);
            percentageOfOptimalActionMatrix(1, (episodeNumber + 1)) = averagePercentageOfOptimalAction;
        else 
            rewardsForAverage = temporaryAverageRewardMatrix(1, (2 : (episodeNumber + 1)));
            averageReward = (sum(rewardsForAverage) / numel(rewardsForAverage));
            averageRewardMatrix(1, (episodeNumber + 1)) = averageReward;
            
            percentageOfOptimalActionForAverage = temporaryPercentageOfOptimalActionMatrix(1, (2 : (episodeNumber + 1)));
            averagePercentageOfOptimalAction = (sum(percentageOfOptimalActionForAverage) / numel(percentageOfOptimalActionForAverage));
            percentageOfOptimalActionMatrix(1, (episodeNumber + 1)) = averagePercentageOfOptimalAction;
        end
    end
return