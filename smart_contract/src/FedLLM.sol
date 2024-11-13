
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { SD59x18, sd } from "@prb/math/src/SD59x18.sol";
import { UD60x18, ud, unwrap } from "@prb/math/src/UD60x18.sol";
import "./ReputationToken.sol";

contract FederatedLearningAggregator {
    address public owner;
    string public modelURI;
    string public modelVersion;
    uint256 public numClients;
    uint256 public totalClients;
    uint256 public trainingRounds;
    uint256 public currentRound;
    ReputationToken public reputationToken;
    mapping(address => bool) public hasSubmitted;
    mapping(address => UD60x18[]) public clientUpdates;
    mapping(address => uint256) public clientSampleSizes;
    UD60x18[] public globalModel;
    address[] public clients;

    mapping(address => uint256) public alignmentScores;
    mapping(address => SD59x18) public consistencyCount;
    mapping(address => uint256) public rewardMultipliers;
    mapping(address => uint256) public stakedAmounts;

    uint256 public baseReward = 100 * 1e18;
    uint256 public consistencyMultiplier = 110;
    uint256 public stakingRequirement = 10 * 1e18;

    event RewardDistributed(address indexed client, uint256 reward);
    event StakeDeposited(address indexed client, uint256 amount);
    event StakeWithdrawn(address indexed client, uint256 amount);
    event ModelSubmitted(address indexed client, UD60x18[] parameters, uint256 sampleSize);
    event ModelAggregated(UD60x18[] globalModel);
    event ResetForNextRound();
    event ResetFederatedLearning();
    event TrainingRoundsSet(uint256 rounds);
    event GlobalModelSet(UD60x18[] newGlobalModel);
    event ModelURIUpdated(string newModelURI, string newVersion);
    event ConsistencyUpdated(address indexed client, int256 consistencyCount);
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    modifier validClient() {
        require(!hasSubmitted[msg.sender], "Client has already submitted");
        require(stakedAmounts[msg.sender] >= stakingRequirement, "Stake requirement not met");
        _;
    }

    constructor(uint256 _totalClients, address _reputationTokenAddress) {
        owner = msg.sender;
        totalClients = _totalClients;
        reputationToken = ReputationToken(_reputationTokenAddress);
    }

    function submitModel(uint256[] memory parameters, uint256 sampleSize) public validClient {
        require(parameters.length > 0, "Parameters cannot be empty");
        require(sampleSize > 0, "Sample size must be greater than zero");

        if (!hasSubmitted[msg.sender]) {
            clients.push(msg.sender);
        }

        UD60x18[] memory scaledParameters = new UD60x18[](parameters.length);
        for (uint256 i = 0; i < parameters.length; i++) {
            scaledParameters[i] = UD60x18.wrap(parameters[i]);
        }

        clientUpdates[msg.sender] = scaledParameters;
        clientSampleSizes[msg.sender] = sampleSize;
        hasSubmitted[msg.sender] = true;
        numClients++;

        emit ModelSubmitted(msg.sender, scaledParameters, sampleSize);

        if (numClients == totalClients) {
            aggregateModels();
        }
    }

    function calculateAlignmentScore(address client) internal returns (uint256) {
        uint256 score = 0;
        SD59x18 scale = sd(1000); // Scaling factor using PRBMath

        // Calculate alignment score using dot product
        for (uint256 i = 0; i < clientUpdates[client].length; i++) {
            score += unwrap(clientUpdates[client][i]) * unwrap(globalModel[i]);
        }

        // Adjust consistency count using scaled score
        SD59x18 scaledScore = sd(int256(score)).div(scale);
        if (scaledScore.unwrap() != 0) {
            consistencyCount[client] = consistencyCount[client].add(scaledScore);
        }

        emit ConsistencyUpdated(client, consistencyCount[client].unwrap());
        return score;
    }



    function aggregateModels() internal {
        uint256 paramCount = clientUpdates[clients[0]].length;
        UD60x18[] memory aggregatedParams = new UD60x18[](paramCount);
        uint256 totalSamples = 0;

        for (uint256 i = 0; i < clients.length; i++) {
            totalSamples += clientSampleSizes[clients[i]];
        }

        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            UD60x18[] memory clientParams = clientUpdates[client];
            uint256 clientSampleSize = clientSampleSizes[client];

            for (uint256 j = 0; j < clientParams.length; j++) {
                UD60x18 weight = ud(clientSampleSize).div(ud(totalSamples));
                aggregatedParams[j] = aggregatedParams[j].add(clientParams[j].mul(weight));
            }

            alignmentScores[client] = calculateAlignmentScore(client);
        }

        globalModel = aggregatedParams;
        emit ModelAggregated(globalModel);

        currentRound++;
        if (currentRound < trainingRounds) {
            resetForNextRound();
        } else {
            emit ResetFederatedLearning();
        }
    }

    function updateMultipliers(address client) external onlyOwner {
        SD59x18 consistency = consistencyCount[client];
        SD59x18 threshold = sd(5 * 1e18); // Use fixed-point representation for threshold

        if (consistency.gte(threshold)) {
            rewardMultipliers[client] = consistencyMultiplier; //110
        } else if (consistency.lt(sd(-5 * 1e18))) {
            rewardMultipliers[client] = 0; // Severe penalty for low consistency
        } else if (consistency.unwrap() < 0) {
            rewardMultipliers[client] = 90; // Reduced multiplier for poor consistency
        } else {
            rewardMultipliers[client] = 100; // Default multiplier
        }
    }



    function distributeRewards() external onlyOwner {
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            uint256 score = alignmentScores[client];
            uint256 multiplier = rewardMultipliers[client] > 0 ? rewardMultipliers[client] : 100;
            uint256 reward = (baseReward * score * multiplier) / 10000;
            reputationToken.mint(client, reward);
            emit RewardDistributed(client, reward);

            alignmentScores[client] = 0;
        }
    }

    function resetForNextRound() internal {
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            hasSubmitted[client] = false;
            delete clientUpdates[client];
            delete clientSampleSizes[client];
        }
        numClients = 0;
        delete clients;
        emit ResetForNextRound();
    }

    function getGlobalModel() public view returns (UD60x18[] memory) {
        return globalModel;
    }

    function setTotalClients(uint256 _totalClients) public onlyOwner {
        totalClients = _totalClients;
    }

    function setTrainingRounds(uint256 _rounds) public onlyOwner {
        require(_rounds > 0, "Training rounds must be greater than zero");
        trainingRounds = _rounds;
        emit TrainingRoundsSet(_rounds);
    }

    function resetFederatedLearning() public onlyOwner {
        delete globalModel;
        numClients = 0;
        currentRound = 0;
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            hasSubmitted[client] = false;
            delete clientUpdates[client];
            delete clientSampleSizes[client];
        }
        delete clients;
        emit ResetFederatedLearning();
    }

    function setGlobalModel(UD60x18[] memory newGlobalModel) public onlyOwner {
        require(newGlobalModel.length > 0, "Global model cannot be empty");
        globalModel = newGlobalModel;
        emit GlobalModelSet(newGlobalModel);
    }

    function updateModelURI(string memory newModelURI, string memory newVersion) public onlyOwner {
        modelURI = newModelURI;
        modelVersion = newVersion;
        emit ModelURIUpdated(newModelURI, newVersion);
    }

    function getModelURI() public view returns (string memory, string memory) {
        return (modelURI, modelVersion);
    }
}
