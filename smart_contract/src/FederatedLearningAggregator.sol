// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { SD59x18, sd, unwrap } from "@prb/math/src/SD59x18.sol";
import { UD60x18, ud }         from "@prb/math/src/UD60x18.sol";
import "./ReputationToken.sol";

contract FederatedLearningAggregator {
    // --- IMMUTABLES & STATE ---
    address public immutable owner;
    ReputationToken public immutable reputationToken;

    string public modelURI;
    string public modelVersion;

    UD60x18[] public globalModel;

    uint32 public totalClients;
    uint32 public trainingRounds;
    uint32 public currentRound;
    uint32 public currentShapleyRound;
    uint32 public lastFairnessCheckRound;
    uint32 public numClients;

    uint256 public bonusRewardPool;

    mapping(address => bool)    public hasSubmitted;
    mapping(address => bytes32) public paramsHashes;       // keccak256(IPFS‑encoded parameters)
    mapping(address => uint256) public clientSampleSizes;
    address[] public clients;

    mapping(address => SD59x18) public alignmentScores;
    mapping(uint32 => mapping(address => int256)) public roundAlignmentScores;

    mapping(address => SD59x18) public consistencyCount;
    mapping(address => uint256) public rewardMultipliers;

    mapping(uint32 => bytes32) public shapleyIPFSHashes;
    mapping(address => uint256) public shapleyValues;

    uint256 public constant baseReward         = 100 * 1e18;
    uint256 public constant stakingRequirement = 10 * 1e18;
    uint256 public constant consistencyMultiplier = 110;

    // --- EVENTS ---
    event ModelSubmitted(
        uint32  indexed round,
        address indexed client,
        bytes32          paramsHash,
        uint256          sampleSize
    );
    event AlignmentScoresUpdated(
        uint32  indexed round,
        address indexed client,
        int256           rawScore
    );
    event ModelAggregated(UD60x18[] newGlobalModel);
    event FairnessCheckTriggered(uint32 round);
    event ResetForNextRound();
    event ResetRound(uint32 nextRound);

    event RewardDistributed(address indexed client, uint256 reward);
    event BonusDistributed(address indexed client, uint256 reward);

    event StakeDeposited(address indexed client, uint256 amount);
    event StakeWithdrawn(address indexed client, uint256 amount);

    event TrainingRoundsSet(uint32 rounds);
    event ModelURIUpdated(string newURI, string newVersion);

    event ConsistencyUpdated(address indexed client, int256 consistencyCount);

    // --- MODIFIERS ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    modifier validClient() {
        require(!hasSubmitted[msg.sender], "Already submitted");
        require(
            stakedAmounts[msg.sender] >= stakingRequirement,
            "Stake requirement not met"
        );
        _;
    }

    mapping(address => uint256) public stakedAmounts;

    // --- CONSTRUCTOR ---
    constructor(uint32 _totalClients, address _reputationTokenAddress) {
        owner            = msg.sender;
        totalClients     = _totalClients;
        reputationToken  = ReputationToken(_reputationTokenAddress);
    }

    // --- 0) ADMIN FUNCTIONS ---
    function setTrainingRounds(uint32 _rounds) external onlyOwner {
        require(_rounds > 0, "Rounds>0");
        trainingRounds = _rounds;
        emit TrainingRoundsSet(_rounds);
    }

    function updateModelURI(string calldata _uri, string calldata _version)
        external
        onlyOwner
    {
        modelURI     = _uri;
        modelVersion = _version;
        emit ModelURIUpdated(_uri, _version);
    }

    // --- 1) STAKING ---
    function depositStake(uint256 amount) external {
        require(amount >= stakingRequirement, "Stake too low");
        stakedAmounts[msg.sender] += amount;
        emit StakeDeposited(msg.sender, amount);
    }

    function withdrawStake(uint256 amount) external {
        require(stakedAmounts[msg.sender] >= amount, "Not enough staked");
        stakedAmounts[msg.sender] -= amount;
        emit StakeWithdrawn(msg.sender, amount);
    }

    // --- 2) UTILITIES ---
    function calculateTotalSamples() internal view returns (SD59x18) {
        SD59x18 total = sd(0);
        for (uint32 i = 0; i < clients.length; i++) {
            total = total.add(
                sd(int256(clientSampleSizes[clients[i]]))
            );
        }
        return total;
    }

    // --- 3) CLIENT SUBMISSION (OFF‑CHAIN STORAGE) ---
    /// @param paramsHash keccak256(abi.encodePacked(IPFS‑encoded JSON))
    function submitModel(bytes32 paramsHash, uint256 sampleSize)
        external
        validClient
    {
        require(sampleSize > 0, "Samples>0");

        hasSubmitted[msg.sender]       = true;
        paramsHashes[msg.sender]       = paramsHash;
        clientSampleSizes[msg.sender]  = sampleSize;
        clients.push(msg.sender);
        numClients += 1;

        emit ModelSubmitted(currentRound, msg.sender, paramsHash, sampleSize);
    }

    // --- 4) ON‑CHAIN ALIGNMENT SCORE ---
    /// @notice Runs dot(gᵢ, g_global) on‑chain with PRB‑Math
    function calculateAlignmentScore(
        address    client,
        UD60x18[] calldata params
    ) public {
        require(hasSubmitted[client], "No submission");
        require(
            keccak256(abi.encodePacked(params)) == paramsHashes[client],
            "Param hash mismatch"
        );

        SD59x18 totalSamples = calculateTotalSamples();
        require(totalSamples.unwrap() > 0, "Zero total samples");

        uint256 L = params.length;
        require(L == globalModel.length, "Length mismatch");

        SD59x18 score = sd(0);
        for (uint256 i; i < L; i++) {
            int256 a = int256(unwrap(params[i]));
            int256 b = int256(unwrap(globalModel[i]));
            score = score.add(sd(a).mul(sd(b)));
        }

        // scale by sampleSize / totalSamples
        score = score.mul(
            sd(int256(clientSampleSizes[client]))
        ).div(totalSamples);

        alignmentScores[client] = score;
        roundAlignmentScores[currentRound][client] = unwrap(score);
        emit AlignmentScoresUpdated(currentRound, client, unwrap(score));
    }

    /// @notice Helper to calculate for all clients in one go
    function calculateAlignmentScore_All(UD60x18[][] calldata allParams)
        external
    {
        require(allParams.length == clients.length, "Param count mismatch");
        for (uint32 i = 0; i < clients.length; i++) {
            calculateAlignmentScore(clients[i], allParams[i]);
        }
    }

    // --- 5) CONSISTENCY MULTIPLIERS ---
    function updateAllMultipliers() public {
        for (uint32 i = 0; i < clients.length; i++) {
            address c = clients[i];
            SD59x18 cnt = consistencyCount[c];
            SD59x18 hi = sd(10 * 1e18);
            SD59x18 lo = sd(5 * 1e18);
            SD59x18 pen = sd(-5 * 1e18);

            uint256 m;
            if (cnt.gte(hi)) {
                m = consistencyMultiplier + 10;
            } else if (cnt.gte(lo)) {
                m = consistencyMultiplier;
            } else if (cnt.lt(pen)) {
                m = 0;
            } else if (cnt.unwrap() < 0) {
                m = 90;
            } else {
                m = 100;
            }

            rewardMultipliers[c] = m;
            emit ConsistencyUpdated(c, cnt.unwrap());
        }
    }

    // --- 6) ON‑CHAIN AGGREGATION (FedAvg) ---
    /// @notice Owner pushes all client arrays in one transaction
    function aggregateModels(
        UD60x18[][] calldata allParams,
        uint256[]    calldata sampleSizes
    ) external onlyOwner {
        uint32 n = uint32(allParams.length);
        require(n == clients.length, "Count mismatch");

        SD59x18 total = sd(0);
        for (uint32 i; i < n; i++) {
            total = total.add(sd(int256(sampleSizes[i])));
        }
        require(total.unwrap() > 0, "No data");

        uint256 M = allParams[0].length;
        UD60x18[] memory agg = new UD60x18[](M);

        for (uint32 i; i < n; i++) {
            require(
                keccak256(abi.encodePacked(allParams[i])) == paramsHashes[clients[i]],
                "Integrity fail"
            );
            require(
                sampleSizes[i] == clientSampleSizes[clients[i]],
                "Size mismatch"
            );

            UD60x18 w = ud(sampleSizes[i])
                .div(ud(uint256(total.unwrap())));

            for (uint256 j; j < M; j++) {
                agg[j] = agg[j].add(allParams[i][j].mul(w));
            }
        }

        delete globalModel;
        for (uint256 k; k < M; k++) {
            globalModel.push(agg[k]);
        }

        emit ModelAggregated(agg);
    }

    // --- 7) OFF‑CHAIN FAIRNESS UPDATE ---
    function updateFairnessData(
        bytes32             ipfsHash,
        bytes32             integrityHash,
        address[] calldata  addrs,
        uint256[] calldata  vals
    ) external onlyOwner {
        require(addrs.length == vals.length, "Len mismatch");
        shapleyIPFSHashes[currentRound] = ipfsHash;
        for (uint256 i; i < addrs.length; i++) {
            shapleyValues[addrs[i]] = vals[i];
        }
        currentShapleyRound++;
        emit FairnessCheckTriggered(currentRound);
        lastFairnessCheckRound = currentRound;
    }

    // --- 8) BONUS & BASE REWARDS ---
    function allocateToBonusPool(uint256 amount) internal {
        bonusRewardPool += amount;
    }

    function fundBonusPool(uint256 amount) external onlyOwner {
        require(amount > 0, "Must be >0");
        bonusRewardPool += amount;
    }

    /// @notice Distribute per‑round on‑chain rewards
    function baseDistributeRewards() external {
        for (uint32 i = 0; i < clients.length; i++) {
            address c = clients[i];
            SD59x18 score = alignmentScores[c];
            SD59x18 mult  = rewardMultipliers[c] > 0
                ? sd(int256(rewardMultipliers[c]))
                : sd(100);

            SD59x18 r = sd(int256(baseReward))
                .mul(score)
                .mul(mult)
                .div(sd(10000));

            uint256 reward = r.unwrap() > 0 ? uint256(r.unwrap()) : 0;
            reputationToken.mint(c, reward);
            emit RewardDistributed(c, reward);
            alignmentScores[c] = sd(0);
        }
    }

    function distributeBonusRewardsAfterFairness() external onlyOwner {
        require(currentShapleyRound > lastFairnessCheckRound, "No new fairness");
        uint256 tot;
        for (uint32 i = 0; i < clients.length; i++) {
            tot += shapleyValues[clients[i]];
        }
        require(tot > 0, "No shapley data");

        for (uint32 i = 0; i < clients.length; i++) {
            address c = clients[i];
            uint256 share = (bonusRewardPool * shapleyValues[c]) / tot;
            if (share > 0) {
                reputationToken.mint(c, share);
                emit BonusDistributed(c, share);
            }
        }
        bonusRewardPool = 0;
    }

    // --- 9) RESET FOR NEXT ROUND ---
    function resetForNextRound() external onlyOwner {
        for (uint32 i; i < clients.length; i++) {
            delete hasSubmitted[clients[i]];
            delete paramsHashes[clients[i]];
            delete clientSampleSizes[clients[i]];
            delete alignmentScores[clients[i]];
            delete rewardMultipliers[clients[i]];
            delete consistencyCount[clients[i]];
        }
        delete clients;
        numClients            = 0;
        emit ResetForNextRound();
    }

    function resetRound() external onlyOwner {
        currentRound += 1;
        emit ResetRound(currentRound);
    }

    // --- 10) ACCESSORS ---
    function getGlobalModel() external view returns (UD60x18[] memory) {
        return globalModel;
    }

    function setTotalClients(uint32 _tc) external onlyOwner {
        totalClients = _tc;
    }
}
