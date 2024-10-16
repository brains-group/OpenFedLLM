// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearningAggregator {
    address public owner;
    string public modelURI;
    string public modelVersion;
    uint256 public numClients;
    uint256 public totalClients;
    uint256 public trainingRounds;
    uint256 public currentRound;
    mapping(address => bool) public hasSubmitted;
    mapping(address => uint256[]) public clientUpdates;
    mapping(address => uint256) public clientSampleSizes;
    uint256[] public globalModel;

    event ModelSubmitted(address indexed client, uint256[] parameters, uint256 sampleSize);
    event ModelAggregated(uint256[] globalModel);
    event ResetForNextRound();
    event ResetFederatedLearning();
    event TrainingRoundsSet(uint256 rounds);
    event GlobalModelSet(uint256[] newGlobalModel);
    event ModelURIUpdated(string newModelURI, string newVersion);
    //event ModelURIRetrieved(address indexed client, string modelURI, string modelVersion);

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

        if (numClients == totalClients) {
            aggregateModels();
        }
    }

    function aggregateModels() internal {
        uint256 paramCount = clientUpdates[msg.sender].length;
        uint256[] memory aggregatedParams = new uint256[](paramCount);
        uint256 totalSamples = 0;

        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // Replace with actual client address iteration in real implementation
            totalSamples += clientSampleSizes[client];
        }

        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // Replace with actual client address iteration in real implementation
            uint256[] memory clientParams = clientUpdates[client];
            uint256 clientSampleSize = clientSampleSizes[client];

            for (uint256 j = 0; j < clientParams.length; j++) {
                aggregatedParams[j] += (clientParams[j] * clientSampleSize) / totalSamples;
            }
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

    function resetForNextRound() internal {
        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // Replace with actual client address iteration in real implementation
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

    function setTrainingRounds(uint256 _rounds) public onlyOwner {
        require(_rounds > 0, "Training rounds must be greater than zero");
        trainingRounds = _rounds;
        emit TrainingRoundsSet(_rounds);
    }

    function resetFederatedLearning() public onlyOwner {
        delete globalModel;
        numClients = 0;
        currentRound = 0;
        for (uint256 i = 0; i < totalClients; i++) {
            address client = msg.sender; // Replace with actual client address iteration in real implementation
            hasSubmitted[client] = false;
            delete clientUpdates[client];
            delete clientSampleSizes[client];
        }
        emit ResetFederatedLearning();
    }

    function setGlobalModel(uint256[] memory newGlobalModel) public onlyOwner {
        require(newGlobalModel.length > 0, "Global model cannot be empty");
        globalModel = newGlobalModel;
        emit GlobalModelSet(newGlobalModel);
    }

    // Allows the owner to update the model URI and version
    function updateModelURI(string memory newModelURI, string memory newVersion) public onlyOwner {
        modelURI = newModelURI;
        modelVersion = newVersion;
        emit ModelURIUpdated(newModelURI, newVersion);
    }

    // Allows clients to retrieve the model URI and version
    function getModelURI() public view returns (string memory, string memory) {
        //emit ModelURIRetrieved(msg.sender, modelURI, modelVersion);
        return (modelURI, modelVersion);
    }
}
