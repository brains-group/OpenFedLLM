// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../src/FedLLM.sol";
import { UD60x18, ud, unwrap } from "@prb/math/src/UD60x18.sol";
import "forge-std/console.sol";

contract FederatedLearningAggregatorTest is Test {
    
    FederatedLearningAggregator aggregator;
    address[] clients;
    function setUp() public {
            // Deploy the aggregator contract with 3 clients
        ReputationToken token = new ReputationToken();
        address reputationTokenAddress = address(token);

        // Deploy the aggregator and pass the token
        aggregator = new FederatedLearningAggregator(3, reputationTokenAddress);

        // Transfer ownership of the token to the aggregator
        token.transferOwnership(address(aggregator));

        // Ensure the ownership transfer
        assertEq(token.owner(), address(aggregator), "Token ownership mismatch");

        // Mock client addresses
        clients.push(address(0x1000)); // Client 1
        clients.push(address(0x2000)); // Client 2
        clients.push(address(0x3000)); // Client 3

        // Stake for each client
        vm.prank(clients[0]);
        aggregator.depositStake(10 * 1e18);
        console.log("Client 1 staked:", aggregator.stakedAmounts(clients[0]));

        vm.prank(clients[1]);
        aggregator.depositStake(10 * 1e18);
        console.log("Client 2 staked:", aggregator.stakedAmounts(clients[1]));

        vm.prank(clients[2]);
        aggregator.depositStake(10 * 1e18);
        console.log("Client 3 staked:", aggregator.stakedAmounts(clients[2]));

        // Validate setup
        assertEq(aggregator.totalClients(), 3, "Total clients mismatch");

    }

    function testBasicLog() public {
        console.log("This is a basic test log");
        assertTrue(true);
    }
    function testSubmitAndAggregate_small_module() public {
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

        // Retrieve the aggregated global model 
        UD60x18[] memory globalModel = aggregator.getGlobalModel();
        
        // Assert that the aggregated global model matches our expected values
        assertApproxEqAbs(unwrap(globalModel[0]), 1.8e18, 0, "Parameter 0 aggregated incorrectly"); 
        assertApproxEqAbs(unwrap(globalModel[1]), 2.88e18, 0, "Parameter 1 aggregated incorrectly");
    }

    function testSubmitAndAggregateLargeParameters_large_module() public {
        // Define larger parameter arrays for each client
        uint256[] memory parameters1 = new uint256[](10);  
        uint256[] memory parameters2 = new uint256[](10);
        uint256[] memory parameters3 = new uint256[](10);     

        // Client 1 submits [1.1, 2.2, 3.3, 4.4, ..., 10.0] with sample size 20
        for (uint256 i = 0; i < 10; i++) {
            parameters1[i] = ud((i + 1) * 1.1e18).unwrap();
        }
        vm.prank(clients[0]);
        aggregator.submitModel(parameters1, 20);

        // Client 2 submits [2.1, 3.2, 4.3, 5.4, ..., 11.0] with sample size 30
        for (uint256 i = 0; i < 10; i++) {
            parameters2[i] = ud((i + 1) * 2.1e18).unwrap();
        }
        vm.prank(clients[1]);
        aggregator.submitModel(parameters2, 30);

        // Client 3 submits [3.1, 4.2, 5.3, 6.4, ..., 12.0] with sample size 50
        for (uint256 i = 0; i < 10; i++) {
            parameters3[i] = ud((i + 1) * 3.1e18).unwrap();
        }
        vm.prank(clients[2]);
        aggregator.submitModel(parameters3, 50);

        // Calculate expected weighted averages manually for verification 
        //The expected weighted average results for all 10 parameters are 
        //[2.4, 4.8, 7.2, 9.6, 12.0, 14.4, 16.8, 19.2, 21.6, 24.0]

        // Retrieve the aggregated global model (should be automatically aggregated)
        UD60x18[] memory globalModel = aggregator.getGlobalModel();

        // Assert the aggregated values for All the parameters to demonstrate correctness
        assertApproxEqAbs(unwrap(globalModel[0]), 2.4e18, 1e14, "Parameter 0 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[1]), 4.8e18, 1e14, "Parameter 1 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[2]), 7.2e18, 1e14, "Parameter 2 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[3]), 9.6e18, 1e14, "Parameter 3 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[4]), 12.0e18, 1e14, "Parameter 4 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[5]), 14.4e18, 1e14, "Parameter 5 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[6]), 16.8e18, 1e14, "Parameter 6 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[7]), 19.2e18, 1e14, "Parameter 7 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[8]), 21.6e18, 1e14, "Parameter 8 aggregated incorrectly");
        assertApproxEqAbs(unwrap(globalModel[9]), 24.0e18, 1e14, "Parameter 9 aggregated incorrectly");
    }
}
