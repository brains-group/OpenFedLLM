// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { SD59x18, sd } from "@prb/math/src/SD59x18.sol";
import { UD60x18, ud, unwrap } from "@prb/math/src/UD60x18.sol";

contract FederatedLearningAggregator {
    address public owner;
    string public modelURI;
    string public modelVersion;
    uint256 public numClients;
    uint256 public totalClients;
    uint256 public trainingRounds;
    uint256 public currentRound;
    mapping(address => bool) public hasSubmitted;
    mapping(address => UD60x18[]) public clientUpdates; // Store parameters as UD60x18[]
    mapping(address => uint256) public clientSampleSizes;
    UD60x18[] public globalModel; // Store global model as UD60x18[]
    address[] public clients;

    event ModelSubmitted(address indexed client, UD60x18[] parameters, uint256 sampleSize);
    event ModelAggregated(UD60x18[] globalModel);
    event ResetForNextRound();
    event ResetFederatedLearning();
    event TrainingRoundsSet(uint256 rounds);
    event GlobalModelSet(UD60x18[] newGlobalModel);
    event ModelURIUpdated(string newModelURI, string newVersion);

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

    // Submit model parameters, scaling the input if not done by the client
    function submitModel(uint256[] memory parameters, uint256 sampleSize) public validClient {
        require(parameters.length > 0, "Parameters cannot be empty");
        require(sampleSize > 0, "Sample size must be greater than zero");

        if (!hasSubmitted[msg.sender]) {
            clients.push(msg.sender);
        }
        
        // No need to convert again if the value is already scaled by clients
        UD60x18[] memory scaledParameters = new UD60x18[](parameters.length);
        for (uint256 i = 0; i < parameters.length; i++) {
            // Instead of converting using `ud()`, assume it's already scaled
            scaledParameters[i] = UD60x18.wrap(parameters[i]); // Directly wrap the uint256 as UD60x18 without multiplying again
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

    function aggregateModels() internal {
        uint256 paramCount = clientUpdates[clients[0]].length;
        UD60x18[] memory aggregatedParams = new UD60x18[](paramCount);
        uint256 totalSamples = 0;

        // Calculate the total sample size
        for (uint256 i = 0; i < clients.length; i++) {
            totalSamples += clientSampleSizes[clients[i]];
        }

        // Aggregate parameters from all clients
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            UD60x18[] memory clientParams = clientUpdates[client];
            uint256 clientSampleSize = clientSampleSizes[client];

            for (uint256 j = 0; j < clientParams.length; j++) {
                // Perform weighted aggregation using PRB-Math
                UD60x18 weight = ud(clientSampleSize).div(ud(totalSamples));
                aggregatedParams[j] = aggregatedParams[j].add(clientParams[j].mul(weight));
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
