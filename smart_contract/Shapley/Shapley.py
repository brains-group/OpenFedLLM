import json
import time
import ipfshttpclient
from web3 import Web3

# Load contract ABI
try:
    with open('../out/FedLLM.sol/FederatedLearningAggregator.json', 'r') as abi_file:
        abi = json.load(abi_file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Error loading ABI: {e}")
    exit(1)

# Web3 and Contract Configuration
INFURA_URL = "http://127.0.0.1:8545"
PRIVATE_KEY = "PRIVATE_KEY"
CONTRACT_ADDRESS = "CONTRACT_ADDRESS"

web3 = Web3(Web3.HTTPProvider(INFURA_URL))
if not web3.isConnected():
    print("Failed to connect to Ethereum node.")
    exit(1)

account = web3.eth.account.from_key(PRIVATE_KEY)
contract = web3.eth.contract(address=Web3.toChecksumAddress(CONTRACT_ADDRESS), abi=abi)

# Contributions tracking
contributions_over_rounds = {}

# Compute cumulative Shapley values
def compute_cumulative_shapley_values(contributions):
    cumulative = {client: sum(scores) for client, scores in contributions.items()}
    total = sum(cumulative.values())
    if total == 0:
        raise ValueError("Total contributions cannot be zero.")
    return {client: round((value / total) * 1e18) for client, value in cumulative.items()}

# Save to IPFS
def save_to_ipfs(data):
    try:
        with ipfshttpclient.connect() as client:
            cid = client.add_json(data)
            print(f"Data uploaded to IPFS with CID: {cid}")
            return cid
    except Exception as e:
        print(f"Error uploading to IPFS: {e}")
        exit(1)

# Compute integrity hash
def compute_integrity_hash(clients, values):
    data = {"clients": clients, "values": values}
    return web3.keccak(text=json.dumps(data)).hex()

# Update Shapley values on-chain
def update_shapley_values_onchain(cid, clients, values, hash):
    nonce = web3.eth.getTransactionCount(account.address)
    try:
        gas_price = web3.eth.gas_price
        tx = contract.functions.updateFairnessData(cid, hash, clients, values).buildTransaction({
            "from": account.address,
            "nonce": nonce,
            "gasPrice": gas_price,
            "gas": 3000000
        })
        signed_tx = web3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        print(f"Transaction sent with hash: {tx_hash.hex()}")
    except Exception as e:
        print(f"Error sending transaction: {e}")
        exit(1)

# Handle events
def handle_alignment_score_event(event):
    try:
        round_number = event['args']['round']
        client = event['args']['client']
        score = event['args']['score']
        print(f"Round {round_number}: Client {client} - Alignment Score {score}")

        if client not in contributions_over_rounds:
            contributions_over_rounds[client] = []
        contributions_over_rounds[client].append(score)
    except Exception as e:
        print(f"Error handling alignment score event: {e}")

def handle_fairness_check_triggered(event):
    try:
        round_number = event['args']['round']
        print(f"Fairness check triggered for round: {round_number}")

        cumulative = compute_cumulative_shapley_values(contributions_over_rounds)
        clients, values = list(cumulative.keys()), list(cumulative.values())
        cid = save_to_ipfs({"round": round_number, "values": cumulative})
        hash = compute_integrity_hash(clients, values)
        update_shapley_values_onchain(cid, clients, values, hash)
    except Exception as e:
        print(f"Error handling fairness check: {e}")

# Event loop
def log_loop(event_filters, poll_interval):
    while True:
        for event_filter, handler in event_filters:
            for event in event_filter.get_new_entries():
                handler(event)
        time.sleep(poll_interval)

# Event filters
filters = [
    (contract.events.AlignmentScoresUpdated.createFilter(fromBlock='latest'), handle_alignment_score_event),
    (contract.events.FairnessCheckTriggered.createFilter(fromBlock='latest'), handle_fairness_check_triggered)
]

# Run loop
log_loop(filters, 2)
