// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/FedLLM.sol";
import { UD60x18, ud, unwrap } from "@prb/math/src/UD60x18.sol";

contract FederatedLearningAggregatorTest is Test {
    FederatedLearningAggregator aggregator;
    address[] clients;

    function setUp() public {
        // Deploy the contract with a total of 3 clients
        aggregator = new FederatedLearningAggregator(3);

        // Create a list of mock client addresses
        clients.push(address(0x1000)); // Mock client 1
        clients.push(address(0x2000)); // Mock client 2
        clients.push(address(0x3000)); // Mock client 3
    }

    function testSubmitAndAggregate() public {
        // Fake data for three clients
        // Client 1 submits [1.2, 2.5] with sample size 10
        uint256[] memory parameters1 = new uint256[](2);
        parameters1[0] = ud(1.2e18).unwrap();//1200000000000000000; // Equivalent to 1.2 in UD60x18
        parameters1[1] = ud(2.5e18).unwrap(); // Equivalent to 2.5 in UD60x18
        vm.prank(clients[0]); // Simulate client 1 submitting
        aggregator.submitModel(parameters1, 10);

        // Client 2 submits [2.2, 3.1] with sample size 15
        uint256[] memory parameters2 = new uint256[](2);
        parameters2[0] = ud(2.2e18).unwrap(); // Equivalent to 2.2 in UD60x18
        parameters2[1] = ud(3.1e18).unwrap(); // Equivalent to 3.1 in UD60x18
        vm.prank(clients[1]); // Simulate client 2 submitting
        aggregator.submitModel(parameters2, 15);

        // Client 3 submits [1.8, 2.9] with sample size 25
        uint256[] memory parameters3 = new uint256[](2);
        parameters3[0] = ud(1.8e18).unwrap(); // Equivalent to 1.8 in UD60x18
        parameters3[1] = ud(2.9e18).unwrap(); // Equivalent to 2.9 in UD60x18
        vm.prank(clients[2]); // Simulate client 3 submitting
        aggregator.submitModel(parameters3, 25);

        // Check the aggregated result
        // In this example, the weight of each parameter is:
        // Client 1 weight = 10 / 50 = 0.2
        // Client 2 weight = 15 / 50 = 0.3
        // Client 3 weight = 25 / 50 = 0.5
        // Expected weighted average for parameter 0:
        // (1.2 * 0.2) + (2.2 * 0.3) + (1.8 * 0.5) = 1.8
        // Expected weighted average for parameter 1:
        // (2.5 * 0.2) + (3.1 * 0.3) + (2.9 * 0.5) = 2.88

        // Retrieve the aggregated global model (should be automatically aggregated)
        UD60x18[] memory globalModel = aggregator.getGlobalModel();
        
        // Assert that the aggregated global model matches our expected values
        assertApproxEqAbs(unwrap(globalModel[0]), 1.8e18, 0, "Parameter 0 aggregated incorrectly"); // Allow for minor precision loss
        assertApproxEqAbs(unwrap(globalModel[1]), 2.88e18, 0, "Parameter 1 aggregated incorrectly");
    }
}
