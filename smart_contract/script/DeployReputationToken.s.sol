// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "forge-std/Script.sol";
import "../src/ReputationToken.sol";

contract DeployReputationToken is Script {
    function run() external {
        // Start broadcasting the transaction to the blockchain
        vm.startBroadcast();

        // Deploy the ReputationToken contract
        ReputationToken token = new ReputationToken();

        // Log the deployed token address
        console.log("ReputationToken deployed at:", address(token));

        // Stop broadcasting the transaction
        vm.stopBroadcast();
    }
}
