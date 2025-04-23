import random
import torch
import json
from web3 import Web3
from ..smart_contract.IPFS.upload_parameters import upload_params_to_ipfs

# === On-Chain Contract Setup ===
INFURA_URL       = "http://127.0.0.1:8545"
CONTRACT_ADDRESS = "ContractAddress"
ABI_PATH         = "../smart_contract/out/FedLLM.sol/FederatedLearningAggregator.json"

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
with open(ABI_PATH) as f:
    abi = json.load(f)
contract = w3.eth.contract(
    address=Web3.toChecksumAddress(CONTRACT_ADDRESS),
    abi=abi
)


def get_clients_this_round(fed_args, round_idx):
    """
    Select clients for this round.
    """
    if fed_args.fed_alg.startswith('local'):
        return [int(fed_args.fed_alg[-1])]
    if fed_args.num_clients <= fed_args.sample_clients:
        return list(range(fed_args.num_clients))
    random.seed(round_idx)
    return sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))


def global_aggregate(*args, **kwargs):
    """
    Stub: Aggregation is now handled on-chain via IPFS and smart contracts.
    Calling this will raise an error to prevent local aggregation.
    """
    raise RuntimeError(
        "Off-chain aggregation enabled: call on-chain FederatedLearningAggregator instead."
    )


def fetch_global_model(global_dict):
    """
    Pull the updated global model from-chain and deserialize back into tensors.
    """
    # raw list of uint256 (UD60x18-scaled) from contract
    raw = contract.functions.getGlobalModel().call()
    keys = list(global_dict.keys())
    for i, key in enumerate(keys):
        # scale back down by 1e18
        val = raw[i] / 1e18
        global_dict[key] = torch.tensor(val).reshape(global_dict[key].shape)
    return global_dict
