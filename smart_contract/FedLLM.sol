// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearningAggregator {
    address public owner;
    uint256 public numClients;
    uint256 public totalClients;
    mapping(address => bool) public hasSubmitted;
    mapping(address => uint256[]) public clientUpdates;
    mapping(address => uint256) public clientSampleSizes; // New: stores sample size for each client
    uint256[] public globalModel;

    event ModelSubmitted(address indexed client, uint256[] parameters, uint256 sampleSize);
    event ModelAggregated(uint256[] globalModel);
    event ResetForNextRound();

    constructor(uint256 _totalClients) {
        owner = msg.sender;
        totalClients = _totalClients;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not authorized");
        _;
    }

    modifier validClient() {
        require(!hasSubmitted[msg.sender], "Client has already submitted");
        _;
    }

    function submitModel(uint256[] memory parameters, uint256 sampleSize) public validClient {
        require(parameters.length > 0, "Parameters cannot be empty");
        require(sampleSize > 0, "Sample size must be greater than zero");

        clientUpdates[msg.sender] = parameters;
        clientSampleSizes[msg.sender] = sampleSize;
        hasSubmitted[msg.sender] = true;
        numClients++;

        emit ModelSubmitted(msg.sender, parameters, sampleSize);

        // If all clients have submitted, trigger aggregation
        if (numClients == totalClients) {
            aggregateModels();
        }
    }

    function aggregateModels() internal {
        uint256 paramCount = clientUpdates[msg.sender].length;
        uint256[] memory aggregatedParams = new uint256[](paramCount);
        uint256 totalSamples = 0;

        // Calculate the total number of samples
        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // iterate over actual client addresses in real implementation
            totalSamples += clientSampleSizes[client];
        }

        // Weighted average of parameters
        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // iterate over actual client addresses in real implementation
            uint256[] memory clientParams = clientUpdates[client];
            uint256 clientSampleSize = clientSampleSizes[client];

            for (uint256 j = 0; j < clientParams.length; j++) {
                aggregatedParams[j] += (clientParams[j] * clientSampleSize) / totalSamples;
            }
        }

        globalModel = aggregatedParams;
        emit ModelAggregated(globalModel);

        // Reset for the next round
        resetForNextRound();
    }

    function resetForNextRound() internal {
        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // iterate over actual client addresses in real implementation
            hasSubmitted[client] = false;
            delete clientUpdates[client];
            delete clientSampleSizes[client];
        }
        numClients = 0;
        
        emit ResetForNextRound();
    }

    function getGlobalModel() public view returns (uint256[] memory) {
        return globalModel;
    }

    function setTotalClients(uint256 _totalClients) public onlyOwner {
        totalClients = _totalClients;
    }
}
