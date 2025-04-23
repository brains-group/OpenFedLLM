import time
import json
from web3 import Web3
import ipfshttpclient

# === Configuration ===
INFURA_URL       = "http://127.0.0.1:8545"
PRIVATE_KEY      = "PRIVATE_KEY"
CONTRACT_ADDRESS = "ContractAddress"
ABI_PATH         = "../out/FedLLM.sol/FederatedLearningAggregator.json"
IPFS_API         = "/ip4/127.0.0.1/tcp/5001/http"
CID_MAP_PATH     = "cid_map.json"
POLL_INTERVAL    = 10  # seconds

# === Setup Web3 & Contract ===
w3 = Web3(Web3.HTTPProvider(INFURA_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)
with open(ABI_PATH) as f:
    abi = json.load(f)
contract = w3.eth.contract(address=Web3.toChecksumAddress(CONTRACT_ADDRESS), abi=abi)

# === Setup IPFS ===
ipfs = ipfshttpclient.connect(IPFS_API)

# === Load paramsHash → CID map ===
with open(CID_MAP_PATH) as f:
    cid_map = json.load(f)

def fetch_submissions(round_number, total_clients):
    """Wait until all clients have submitted for this round."""
    seen = {}
    print(f"[Manager] Waiting for {total_clients} submissions in round {round_number}")
    while len(seen) < total_clients:
        events = contract.events.ModelSubmitted.get_logs(
            fromBlock='latest',
            argument_filters={'round': round_number}
        )
        for ev in events:
            seen[ev['args']['client']] = ev['args']
        print(f"  ↳ {len(seen)}/{total_clients} submissions")
        time.sleep(POLL_INTERVAL)
    return list(seen.values())

def fetch_params(cid: str):
    """Retrieve JSON array of uint256 values from IPFS."""
    return ipfs.get_json(cid)

def send_tx(func, *args, gas=1_000_000):
    """Helper to build, sign, and send a transaction."""
    tx = func(*args).buildTransaction({
        'from': account.address,
        'nonce': w3.eth.getTransactionCount(account.address),
        'gasPrice': w3.eth.gas_price,
        'gas': gas
    })
    signed = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    return w3.eth.send_raw_transaction(signed.rawTransaction)

def manage_round(round_number):
    total_clients = contract.functions.totalClients().call()
    submissions = fetch_submissions(round_number, total_clients)

    # Collect parameters and sampleSizes
    all_params = []
    sample_sizes = []
    for sub in submissions:
        ph = sub['paramsHash']
        cid = cid_map.get(ph)
        if not cid:
            raise ValueError(f"No CID mapping for hash {ph}")
        raw = fetch_params(cid)
        all_params.append([int(x) for x in raw])
        sample_sizes.append(sub['sampleSize'])

    # Alignment scoring
    print(f"[Manager] Running calculateAlignmentScore_All for round {round_number}")
    send_tx(contract.functions.calculateAlignmentScore_All, all_params, gas=3_000_000)

    # Multiplier update
    print("[Manager] Running updateAllMultipliers")
    send_tx(contract.functions.updateAllMultipliers, gas=200_000)

    # Base reward distribution
    print("[Manager] Running baseDistributeRewards")
    send_tx(contract.functions.baseDistributeRewards, gas=500_000)

    # Model aggregation
    print("[Manager] Running aggregateModels")
    send_tx(contract.functions.aggregateModels, all_params, sample_sizes, gas=5_000_000)

    # Next round
    print("[Manager] Advancing round")
    send_tx(contract.functions.resetRound, gas=200_000)

    print(f"[Manager] Completed round {round_number}")

def main():
    current_round = contract.functions.currentRound().call()
    print(f"[Manager] Starting at round {current_round}")
    while True:
        manage_round(current_round)
        current_round += 1
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()