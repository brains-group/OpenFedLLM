import torch
import copy
import json
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from ..smart_contract.IPFS.upload_parameters import upload_to_ipfs_and_submit

class IPFSUploadCallback(TrainerCallback):
    """
    TrainerCallback to upload client parameters to IPFS and call submitModel on-chain
    when local training ends.
    """
    def __init__(self, ipfs_addr, rpc_url, contract_addr, abi_path, sample_size):
        super().__init__()
        self.ipfs_addr     = ipfs_addr
        self.rpc_url       = rpc_url
        self.contract_addr = contract_addr
        self.abi_path      = abi_path
        self.sample_size   = sample_size

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            return
        # Extract and flatten state dict
        sd = get_peft_model_state_dict(model)
        int_params = []
        for key in sorted(sd.keys()):
            arr = sd[key].cpu().numpy().flatten()
            # scale to UD60x18 (1e18)
            ints = (arr * 1e18).astype(int).tolist()
            int_params.extend(ints)
        # Upload to IPFS and on-chain submit
        cid, params_hash = upload_to_ipfs_and_submit(
            self.ipfs_addr,
            self.rpc_url,
            self.contract_addr,
            self.abi_path,
            int_params,
            self.sample_size
        )
        print(f"[IPFS] Uploaded to {cid}, on-chain hash {params_hash}")

# SCAFFOLD callback unchanged
class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)

# Trainer factory

def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer,
                              training_args, local_dataset,
                              formatting_prompts_func, data_collator,
                              global_dict, local_auxiliary, global_auxiliary):
    # Determine sample size for this client
    sample_size = len(local_dataset)

    # Base trainers
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif fed_args.fed_alg in ['fedavg'] or fed_args.fed_alg.startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')

    # Attach IPFS upload callback to run after training
    trainer.add_callback(
        IPFSUploadCallback(
            fed_args.ipfs_addr,
            fed_args.rpc_url,
            fed_args.contract_addr,
            fed_args.abi_path,
            sample_size,
        )
    )
    return trainer


# Trainer subclasses unchanged
class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super().__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            loss += self.mu/2 * torch.norm(param - self.global_state[name])**2
        return (loss, outputs) if return_outputs else loss

class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super().__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)
        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    def get_auxiliary_param(self):
        auxiliary_new = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                aux_new = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                auxiliary_new[name] = aux_new
                auxiliary_delta[name] = aux_new - self.local_auxiliary[name]
        return auxiliary_new, auxiliary_delta

