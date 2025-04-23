import json
import argparse
import ipfshttpclient
from web3 import Web3

# === Configuration ===
INFURA_URL       = "http://127.0.0.1:8545"
PRIVATE_KEY      = "PRIVATE_KEY"
CONTRACT_ADDRESS = "CONTRACT_ADDRESS"
ABI_PATH         = "../out/FedLLM.sol/FederatedLearningAggregator.json"

# === Helpers ===
def load_abi(path: str):
    with open(path) as f:
        return json.load(f)

def upload_params_to_ipfs(params: list, ipfs_address: str = "/ip4/127.0.0.1/tcp/5001/http") -> str:
    """
    Connects to IPFS and stores the JSON-serializable `params`.
    Returns the CID (base58 string).
    """
    client = ipfshttpclient.connect(ipfs_address)
    cid = client.add_json(params)
    client.close()
    return cid

def compute_params_hash(params: list) -> str:
    """
    Computes the keccak256 hash of the packed uint256[] array,
    matching solidity: keccak256(abi.encodePacked(params)).
    """
    w3 = Web3()
    # Ensure plain Python ints (no wrappers)
    return w3.solidityKeccak(['uint256[]'], [params]).hex()

# === Main ===
def main():
    parser = argparse.ArgumentParser(
        description="Upload FL parameters to IPFS and call submitModel()"
    )
    parser.add_argument("params_file", help="JSON file with a list of integers")
    parser.add_argument("sample_size", type=int, help="Number of data samples")
    parser.add_argument(
        "--ipfs", default="/ip4/127.0.0.1/tcp/5001/http",
        help="Multiaddr of IPFS daemon"
    )
    args = parser.parse_args()

    # Load parameters from JSON
    with open(args.params_file) as f:
        params = json.load(f)

    # Upload to IPFS
    cid = upload_params_to_ipfs(params, args.ipfs)
    print(f"Uploaded to IPFS: CID = {cid}")

    # Compute keccak256 hash for submitModel
    params_hash = compute_params_hash(params)
    print(f"paramsHash = {params_hash}")

    # Build and send transaction
    w3 = Web3(Web3.HTTPProvider(INFURA_URL))
    if not w3.isConnected():
        raise RuntimeError("Web3 provider not connected")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    abi     = load_abi(ABI_PATH)
    contract= w3.eth.contract(
        address=Web3.toChecksumAddress(CONTRACT_ADDRESS),
        abi=abi
    )

    nonce     = w3.eth.getTransactionCount(account.address)
    gas_price = w3.eth.gas_price
    tx = contract.functions.submitModel(params_hash, args.sample_size).buildTransaction({
        "from": account.address,
        "nonce": nonce,
        "gasPrice": gas_price,
        "gas": 500_000
    })

    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    print(f"Submitted submitModel(): txHash = {tx_hash.hex()}")

if __name__ == "__main__":
    main()
