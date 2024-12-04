// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "forge-std/console.sol";
import "../src/FedLLM.sol";
import "../src/ReputationToken.sol";

import { SD59x18, sd } from "@prb/math/src/SD59x18.sol";
import { UD60x18, ud, unwrap } from "@prb/math/src/UD60x18.sol";

contract AlignmentScoreTest is Test {
    FederatedLearningAggregator aggregator;
    ReputationToken reputationToken;

    address client1 = address(0x1);
    address client2 = address(0x2);

    function setUp() public {
        // Deploy ReputationToken
        ReputationToken token = new ReputationToken();
        address reputationTokenAddress = address(token);

        // Deploy the aggregator and pass the token
        aggregator = new FederatedLearningAggregator(2, reputationTokenAddress);

        // Transfer ownership of the token to the aggregator
        token.transferOwnership(address(aggregator));

        // Ensure the ownership transfer
        assertEq(token.owner(), address(aggregator), "Token ownership mismatch");

        // Set owner
        vm.prank(address(this));
        aggregator.setTotalClients(2);

        // Fund stakes for the clients
        vm.prank(client1);
        aggregator.depositStake(10 * 1e18);

        vm.prank(client2);
        aggregator.depositStake(10 * 1e18);

    }

    function testCalculateAlignmentScore() public {
        // Mock parameters and sample sizes for the clients
        uint256[] memory client1Params = new uint256[](3);
        uint256[] memory client2Params = new uint256[](3);

        client1Params[0] = 100;
        client1Params[1] = 200;
        client1Params[2] = 300;

        client2Params[0] = 400;
        client2Params[1] = 500;
        client2Params[2] = 600;

        uint256 sampleSize1 = 10;
        uint256 sampleSize2 = 20;

        // Submit models for both clients
        vm.prank(client1);
        aggregator.submitModel(client1Params, sampleSize1);

        vm.prank(client2);
        aggregator.submitModel(client2Params, sampleSize2);

        // Fetch and unwrap the global model
        UD60x18[] memory globalModelRaw = aggregator.getGlobalModel();
        assertEq(globalModelRaw.length, 3, "Global model is empty or unexpected length");
        uint256[] memory globalModel = new uint256[](globalModelRaw.length);
        for (uint256 i = 0; i < globalModelRaw.length; i++) {
            globalModel[i] = unwrap(globalModelRaw[i]);
        }

                // Check alignment scores
        SD59x18 alignmentScore1 = aggregator.alignmentScores(client1);
        SD59x18 alignmentScore2 = aggregator.alignmentScores(client2);

        // Calculate total samples
        SD59x18 totalSamples = sd(int256(sampleSize1 + sampleSize2));

        SD59x18 expectedScore1 = calculateWeightedScore(
            client1Params,
            globalModel,
            sd(int256(sampleSize1)), // Convert sampleSize1 to SD59x18
            totalSamples // Convert totalSamples to SD59x18
        );

        SD59x18 expectedScore2 = calculateWeightedScore(
            client2Params,
            globalModel,
            sd(int256(sampleSize2)), // Convert sampleSize2 to SD59x18
            totalSamples // Convert totalSamples to SD59x18
        );
        console.log("Score1",expectedScore1);
        console.log("Score2",expectedScore2);
        // Validate scores
        assertEq(
            alignmentScore1.unwrap(),
            expectedScore1.unwrap(),
            "Client 1 alignment score mismatch"
        );
        assertEq(
            alignmentScore2.unwrap(),
            expectedScore2.unwrap(),
            "Client 2 alignment score mismatch"
        );




    }

    function calculateWeightedScore(
        uint256[] memory clientParams,
        uint256[] memory globalModel,
        SD59x18 sampleSize,
        SD59x18 totalSamples
    ) internal pure returns (SD59x18) {
        SD59x18 dotProduct = sd(0);

        for (uint256 i = 0; i < clientParams.length; i++) {
            // Convert uint256 values to SD59x18 for signed arithmetic
            SD59x18 clientValue = sd(int256(clientParams[i]));
            SD59x18 globalValue = sd(int256(globalModel[i]));

            // Calculate the dot product
            dotProduct = dotProduct.add(clientValue.mul(globalValue));
        }

        // Apply weighting using sample size and total samples
        return dotProduct.mul(sampleSize).div(totalSamples);
    }



}
