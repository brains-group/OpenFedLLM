// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract ReputationToken is ERC20, Ownable {
    constructor() ERC20("ReputationToken", "REP") Ownable(msg.sender) {
        // Pass the initial owner explicitly to Ownable's constructor
    }

    // Function to mint tokens to clients
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }
}
