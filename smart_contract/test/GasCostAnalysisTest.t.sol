// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import "../src/FederatedLearningAggregator.sol";
import "../src/ReputationToken.sol";

contract GasCostAnalysisTest is Test {
    FederatedLearningAggregator public aggregator;
    ReputationToken public reputationToken;

    address client1 = address(0x1);
    address client2 = address(0x2);
    address client3 = address(0x3);
    address client4 = address(0x4);
    address client5 = address(0x5);

    function setUp() public {
        // Deploy the ReputationToken
        ReputationToken token = new ReputationToken();
        address reputationTokenAddress = address(token);

        // Deploy the FederatedLearningAggregator with the token
        aggregator = new FederatedLearningAggregator(5, reputationTokenAddress);

        // Transfer ownership of the token to the aggregator
        token.transferOwnership(address(aggregator));

        // Ensure initial staking balance for each client
        vm.deal(client1, 10 ether);
        vm.deal(client2, 10 ether);
        vm.deal(client3, 10 ether);
        vm.deal(client4, 10 ether);
        vm.deal(client5, 10 ether);

        // Simulate staking for all clients
        vm.prank(client1);
        aggregator.depositStake(10 ether);
        vm.prank(client2);
        aggregator.depositStake(10 ether);
        vm.prank(client3);
        aggregator.depositStake(10 ether);
        vm.prank(client4);
        aggregator.depositStake(10 ether);
        vm.prank(client5);
        aggregator.depositStake(10 ether);
    }

    function testGasCost() public {
        string memory selectedTest = vm.envString("RUN_TEST");

        if (keccak256(bytes(selectedTest)) == keccak256(bytes("10"))) {
            runGasTest(10);
        } else if (keccak256(bytes(selectedTest)) == keccak256(bytes("100"))) {
            runGasTest(100);
        } else if (keccak256(bytes(selectedTest)) == keccak256(bytes("1000"))) {
            runGasTest(1000);
        } else if (keccak256(bytes(selectedTest)) == keccak256(bytes("10000"))) {
            runGasTest(10000);
        } else if (keccak256(bytes(selectedTest)) == keccak256(bytes("100000"))) {
            runGasTest(100000);
        } else {
            revert("Invalid RUN_TEST value");
        }
    }

    function runGasTest(uint256 size) internal {
        uint256[] memory parameters1 = createParameters(size);
        uint256[] memory parameters2 = createParameters(size);
        uint256[] memory parameters3 = createParameters(size);
        uint256[] memory parameters4 = createParameters(size);
        uint256[] memory parameters5 = createParameters(size);

        // Simulate model submission for all clients
        vm.prank(client1);
        aggregator.submitModel(parameters1, 50);
        vm.prank(client2);
        aggregator.submitModel(parameters2, 40);
        vm.prank(client3);
        aggregator.submitModel(parameters3, 60);
        vm.prank(client4);
        aggregator.submitModel(parameters4, 70);
        vm.prank(client5);
        aggregator.submitModel(parameters5, 30);

        // Aggregate models
        aggregator.aggregateModels();

        // Calculate alignment scores
        aggregator.calculateAlignmentScore_All();

        // Distribute rewards
        aggregator.baseDistributeRewards();
    }

    // Generate test parameters
    function createParameters(uint256 size) public pure returns (uint256[] memory) {
        uint256[] memory parameters = new uint256[](size);
        for (uint256 i = 0; i < size; i++) {
            // Cycle values between 1.0 and 2.0 (1e18 to 2e18)
            parameters[i] = 1e18 + (i % 1e5) * 1e13; // Modulo ensures a repeating pattern
        }
        return parameters;
    }
}
