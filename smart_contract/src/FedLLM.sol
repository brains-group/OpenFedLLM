// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
import "forge-std/console.sol";
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
    uint256 public consistencyCheckInterval = 5;
    uint256 public lastConsistencyCheckRound = 0;
    uint256 public currentShapleyRound = 0;
    ReputationToken public reputationToken;

    uint256 public bonusRewardPool;
    mapping(uint256 => string) public shapleyIPFSCIDs; // Round -> IPFS CID
    mapping(address => uint256) public shapleyValues;
    uint256 public fairnessCheckInterval = 5; // Check every 5 rounds
    uint256 public lastFairnessCheckRound = 0;

    mapping(address => bool) public hasSubmitted;
    mapping(address => UD60x18[]) public clientUpdates;
    mapping(address => uint256) public clientSampleSizes;
    UD60x18[] public globalModel;
    address[] public clients;

    mapping(address => SD59x18) public alignmentScores;
    mapping(address => SD59x18) public consistencyCount;
    mapping(address => uint256) public rewardMultipliers;
    mapping(address => uint256) public stakedAmounts;
    mapping(uint256 => mapping(address => uint256)) public roundAlignmentScores;

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
    event ShapleyValuesUpdated(uint256 round, string cid);
    event BonusDistributed(address indexed client, uint256 reward);
    event FairnessCheckTriggered(uint256 round);
    event AlignmentScoresUpdated(uint256 indexed round, address indexed client, int256 score);

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

    function depositStake(uint256 amount) external {
        //console.log("Client 1 staked:", amount);
        require(amount >= stakingRequirement, "Stake requirement not met");
        stakedAmounts[msg.sender] += amount;
        // console.log("Client 1 staked:", stakedAmounts[msg.sender]);
        emit StakeDeposited(msg.sender, amount);
    }

    function submitModel(uint256[] memory parameters, uint256 sampleSize) public validClient {
            require(parameters.length > 0, "Parameters cannot be empty");
            require(sampleSize > 0, "Sample size must be greater than zero");
            console.log("numClients:", numClients);
            console.log("totalClients:", totalClients);

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

    function calculateAlignmentScore(address client) internal returns (SD59x18) {
        SD59x18 score = sd(0);
        SD59x18 sampleSize = sd(int256(clientSampleSizes[client]));
        SD59x18 totalSamples = sd(0);

        // Calculate the total sample size
        for (uint256 i = 0; i < clients.length; i++) {
            totalSamples = totalSamples.add(sd(int256(clientSampleSizes[clients[i]])));
        }

        // Calculate weighted dot product alignment score
        for (uint256 i = 0; i < clientUpdates[client].length; i++) {
            // Convert UD60x18 values to int256 safely
            int256 clientValue = int256(unwrap(clientUpdates[client][i]));
            int256 globalValue = int256(unwrap(globalModel[i]));

            // Accumulate the dot product
            score = score.add(sd(clientValue).mul(sd(globalValue)));
        }

        // Apply weighting based on sample size
        score = score.mul(sampleSize).div(totalSamples);

        // Adjust consistency count directly
        consistencyCount[client] = consistencyCount[client].add(score);

        // Emit events for transparency
        emit AlignmentScoresUpdated(currentRound, client, score.unwrap());
        emit ConsistencyUpdated(client, consistencyCount[client].unwrap());

        return score;
    }





    /**
     * @dev Accept off-chain computed Shapley values and link to IPFS CID.
     * @param ipfsCid The IPFS CID where Shapley values are stored.
     * @param clientAddresses List of client addresses.
     * @param values List of Shapley values corresponding to the clients.
     */
    function updateFairnessData(string calldata ipfsCid, bytes32 computedHash, address[] calldata clientAddresses, uint256[] calldata values
    ) external onlyOwner {
        require(bytes(ipfsCid).length > 0, "CID cannot be empty");
        require(clientAddresses.length == values.length, "Input length mismatch");

        // Verify the hash for data integrity
        require(
            keccak256(abi.encode(clientAddresses, values)) == computedHash,
            "Hash mismatch"
        );

        // Store the IPFS CID for the current fairness round
        currentShapleyRound++;
        shapleyIPFSCIDs[currentShapleyRound] = ipfsCid;

        // Accumulate the Shapley values
        for (uint256 i = 0; i < clientAddresses.length; i++) {
            shapleyValues[clientAddresses[i]] += values[i];
        }

        emit ShapleyValuesUpdated(currentShapleyRound, ipfsCid);
    }



    /**
     * @dev Retrieve the CID for a specific Shapley computation round.
     * @param round The round number.
     */
    function getShapleyCID(uint256 round) external view returns (string memory) {
        return shapleyIPFSCIDs[round];
    }

    /**
     * @dev Allocate a portion of rewards to the bonus pool.
     */
    function allocateToBonusPool(uint256 amount) internal {
        bonusRewardPool += amount;
    }

    function fundBonusPool(uint256 amount) external onlyOwner {
        require(amount > 0, "Amount must be greater than zero");
        bonusRewardPool += amount;
    }


    function aggregateModels() internal {
        console.log("Entering aggregateModels");
        uint256 paramCount = clientUpdates[clients[0]].length;
        //console.log("Passed");
        UD60x18[] memory aggregatedParams = new UD60x18[](paramCount);
        uint256 totalSamples = 0;

        // Calculate the total sample size
        for (uint256 i = 0; i < clients.length; i++) {
            totalSamples += clientSampleSizes[clients[i]];
        }
        //console.log("Passed");

        // Aggregate parameters from all clients
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            UD60x18[] memory clientParams = clientUpdates[client];
            uint256 clientSampleSize = clientSampleSizes[client];

            for (uint256 j = 0; j < clientParams.length; j++) {
                UD60x18 weight = ud(clientSampleSize).div(ud(totalSamples));
                aggregatedParams[j] = aggregatedParams[j].add(clientParams[j].mul(weight));
            }
            //console.log("Passed");
            

        }
        
        // Update the global model
        globalModel = aggregatedParams;
        emit ModelAggregated(globalModel);

        // Calculate alignment scores after global model aggregation
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            alignmentScores[client] = calculateAlignmentScore(client);
            //console.log("Alignment score for client", client, ":", alignmentScores[client]);
        }
        console.log("Aggregator1 Passed");
        // Automatically distribute rewards
        baseDistributeRewards();
        console.log("Aggregator2 Passed");
        // Check consistency and update multipliers every few rounds
        if (currentRound >= lastConsistencyCheckRound + consistencyCheckInterval) {
            updateAllMultipliers();
            lastConsistencyCheckRound = currentRound;
        }
        // Trigger fairness check if due
        if (currentRound % fairnessCheckInterval == 0) {
            lastFairnessCheckRound = currentRound;
            emit FairnessCheckTriggered(currentRound);
        }

        currentRound++;
        if (currentRound < trainingRounds) {
            resetForNextRound();
        } else {
            emit ResetFederatedLearning();
        }
    }
    

    function updateAllMultipliers() internal {
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            SD59x18 consistency = consistencyCount[client];
            SD59x18 highThreshold = sd(10 * 1e18); // High threshold for boosted multiplier
            SD59x18 lowThreshold = sd(5 * 1e18);   // Standard consistency threshold
            SD59x18 severePenaltyThreshold = sd(-5 * 1e18); // Severe penalty threshold

            // Assign multipliers based on the consistency count
            if (consistency.gte(highThreshold)) {
                rewardMultipliers[client] = consistencyMultiplier + 10; // Boosted multiplier for very high consistency
            } else if (consistency.gte(lowThreshold)) {
                rewardMultipliers[client] = consistencyMultiplier; // Standard consistency multiplier
            } else if (consistency.lt(severePenaltyThreshold)) {
                rewardMultipliers[client] = 0; // Severe penalty for very low consistency
            } else if (consistency.unwrap() < 0) {
                rewardMultipliers[client] = 90; // Reduced multiplier for poor consistency
            } else {
                rewardMultipliers[client] = 100; // Default multiplier
            }

            emit ConsistencyUpdated(client, consistency.unwrap());
        }
    }


    function updateMultipliers(address client) external onlyOwner {
        SD59x18 consistency = consistencyCount[client];
        SD59x18 highThreshold = sd(10 * 1e18); // High threshold for boosted multiplier
        SD59x18 lowThreshold = sd(5 * 1e18);   // Standard consistency threshold
        SD59x18 severePenaltyThreshold = sd(-5 * 1e18); // Severe penalty threshold

        // Assign multipliers based on the consistency count
        if (consistency.gte(highThreshold)) {
            rewardMultipliers[client] = consistencyMultiplier + 10; // Boosted multiplier for very high consistency
        } else if (consistency.gte(lowThreshold)) {
            rewardMultipliers[client] = consistencyMultiplier; // Standard consistency multiplier
        } else if (consistency.lt(severePenaltyThreshold)) {
            rewardMultipliers[client] = 0; // Severe penalty for very low consistency
        } else if (consistency.unwrap() < 0) {
            rewardMultipliers[client] = 90; // Reduced multiplier for poor consistency
        } else {
            rewardMultipliers[client] = 100; // Default multiplier
        }

        emit ConsistencyUpdated(client, consistency.unwrap());
    }




    // Original logic
    function baseDistributeRewards() internal {
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            SD59x18 score = alignmentScores[client];
            SD59x18 multiplier = rewardMultipliers[client] > 0
                ? sd(int256(rewardMultipliers[client]))
                : sd(100);

            // Calculate reward using SD59x18
            SD59x18 scaledBaseReward = sd(int256(baseReward));
            SD59x18 scaledReward = scaledBaseReward.mul(score).mul(multiplier).div(sd(10000));

            // Ensure reward is non-negative and fits in uint256
            uint256 reward = scaledReward.unwrap() > 0 ? uint256(scaledReward.unwrap()) : 0;

            // Mint the reward to the client
            reputationToken.mint(client, reward);
            emit RewardDistributed(client, reward);

            // Reset alignment score for the next round
            alignmentScores[client] = sd(0);
        }
    }




    function distributeBonusRewardsAfterFairness() external onlyOwner {
        require(currentShapleyRound > lastFairnessCheckRound, "No new fairness data");
        uint256 totalShapley = 0;

        // Calculate total Shapley value
        for (uint256 i = 0; i < clients.length; i++) {
            totalShapley += shapleyValues[clients[i]];
        }
        require(totalShapley > 0, "No Shapley values to distribute");

        // Distribute rewards proportionally
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            uint256 clientShare = (bonusRewardPool * shapleyValues[client]) / totalShapley;
            if (clientShare > 0) {
                reputationToken.mint(client, clientShare);
                emit BonusDistributed(client, clientShare);
            }
        }

        bonusRewardPool = 0; // Reset bonus pool
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
