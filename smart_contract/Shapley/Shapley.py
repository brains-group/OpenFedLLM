import json
import ipfshttpclient
from web3 import Web3

# Web3 and Contract Configuration
try:
    with open('../out/FedLLM.sol/FederatedLearningAggregator.json', 'r') as abi_file:
        abi = json.load(abi_file)
except FileNotFoundError:
    print("ABI file not found. Make sure 'contract_abi.json' exists.")
    exit(1)
except json.JSONDecodeError:
    print("Error decoding ABI JSON. Check the file for errors.")
    exit(1)
    
INFURA_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "PRIVATE_KEY"
CONTRACT_ADDRESS = "CONTRACT_ADDRESS"

# Connect to Ethereum Network
web3 = Web3(Web3.HTTPProvider(INFURA_URL))
account = web3.eth.account.from_key(PRIVATE_KEY)
contract = web3.eth.contract(address=Web3.toChecksumAddress(CONTRACT_ADDRESS), abi=abi)

# Example Data for Clients and Contributions
clients = [
    "0xClient1Address",
    "0xClient2Address",
    "0xClient3Address"
]
contributions = [100, 200, 50]  # Example contributions

# Shapley Value Calculation (Simple Weighted Approximation)
def compute_shapley_values(contributions):
    total = sum(contributions)
    return [round((c / total) * 1e18) for c in contributions]  # Scale by 1e18 for Solidity compatibility

# Compute Shapley Values
shapley_values = compute_shapley_values(contributions)

# Save Shapley Values to IPFS
def save_to_ipfs(data):
    with ipfshttpclient.connect() as client:
        cid = client.add_json(data)
        return cid

shapley_data = {
    "round": 1,  
    "values": {clients[i]: shapley_values[i] for i in range(len(clients))}
}
ipfs_cid = save_to_ipfs(shapley_data)
print(f"Shapley data uploaded to IPFS with CID: {ipfs_cid}")

# Interact with Smart Contract
def update_shapley_values_onchain(cid, client_addresses, values):
    # Prepare transaction
    nonce = web3.eth.getTransactionCount(account.address)
    gas_price = web3.eth.gas_price

    # Call the update function
    tx = contract.functions.updateShapleyValues(cid, client_addresses, values).buildTransaction({
        "from": account.address,
        "nonce": nonce,
        "gasPrice": gas_price,
        "gas": 3000000  # Adjust gas limit based on contract complexity
    })

    # Sign and send transaction
    signed_tx = web3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
    print(f"Transaction sent with hash: {tx_hash.hex()}")

# Call the contract function
update_shapley_values_onchain(ipfs_cid, clients, shapley_values)
