import torch
import copy
from trl import SFTTrainer
from transformers import (
    Trainer, TrainerCallback, TrainerState, TrainerControl, logging,
    DataCollatorForLanguageModeling, PreTrainedTokenizer, training_args, modeling_utils
)
from peft import get_peft_model_state_dict, set_peft_model_state_dict


import pandas as pd
import datasets
from datasets import Dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers.file_utils import is_datasets_available


import opacus
from opacus.accountants import RDPAccountant
from prv_accountant import Accountant as PRVAccountant
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from accelerate.optimizer import AcceleratedOptimizer

from utils.dp_utils import DPCallback, DataCollatorForPrivateCausalLanguageModeling, GradSampleModule, create_author_mapping
from utils.dp_sampler import ShuffledAuthorSampler


logger = logging.get_logger(__name__)


def get_fed_local_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary, global_auxiliary, privacy_args, grad_sampled):
    
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
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator
        )
    elif fed_args.fed_alg == 'dp-alg':
        trainer = OpacusDPTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            data_collator=data_collator,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            privacy_args=privacy_args,
            grad_sampled=grad_sampled,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

class SFTTrainerFedProx(SFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(SFTTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD(SFTTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

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
        
        
class OpacusDPTrainer(SFTTrainer):
    """
    Wrapper to modify Huggingface Trainer to:
        (i) remove "loss = loss / self.args.gradient_accumulation_steps" operation in training_step
        as this is already handled by Opacus package.
        (ii) enable author-level DP training by modifing the sampler and the dataloader. In the case
        of sample-level DP, each sample can be represented by a unique author.
        (iii) wrap the optimizer with Opacus' DPOptimizer/DistributedDPOptimizer
    """
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module] = None,
        args = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        privacy_args = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        grad_sampled: bool = False,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        if author_mapping is None:
            author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
                num_samples=len(self.author_mapping),
            )

        # Wrap model in DDP and GradSampleModule
        if args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            logger.info(f"Wrapping the model with DPDDP in distributed training.")
            model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(model)

        # Instantiate privacy accountants
        self.rdp_accountant = RDPAccountant()
        self.prv_accountant = PRVAccountant(
            noise_multiplier=self.privacy_args.noise_multiplier,
            sampling_probability=self.sampling_probability,
            delta=self.privacy_args.target_delta,
            eps_error=0.1,
            max_compositions=self.num_steps
        )

        # Set up callback for accounting and handling grad acc
        self.dp_callback = DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            target_delta=self.privacy_args.target_delta,
            sampling_probability=self.sampling_probability,
            rdp_accountant=self.rdp_accountant,
            prv_accountant=self.prv_accountant
        )
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=[self.dp_callback], **kwargs)

        if not isinstance(model, GradSampleModule) and not grad_sampled:
            model = GradSampleModule(model)
        
        
        self.get_rdp_epsilon = lambda: self.rdp_accountant.get_epsilon(self.privacy_args.target_delta)  # RDP epsilon
        self.get_prv_epsilon = lambda: self.prv_accountant.compute_epsilon(self.state.global_step)[2]

    @property
    def sampling_probability(self) -> float:
        return self.train_args.per_device_train_batch_size * self.train_args.world_size * \
            self.train_args.gradient_accumulation_steps / len(self.author_mapping)

    @property
    def num_steps(self) -> int:
        return int(self.train_args.num_train_epochs * (1 / self.sampling_probability + 1))

    def create_optimizer(self):
        _ = super().create_optimizer()

        if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            optimizer_generator = opacus.optimizers.DistributedDPOptimizer
        else:
            optimizer_generator = opacus.optimizers.DPOptimizer

        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
        )

        return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Compared to the original HF implementation, we have to remove the loss scaling by the number of gradient
        # accumulation steps since opacus scales the gradients accordingly. However, we still need to scale the loss
        # that is returned in order for the logging to work correctly. Hence we scale the loss after the call to 
        # loss.backward()

        if self.use_apex:
            raise NotImplementedError("DP currently doesn't support this")
        else:
            loss.backward()

        return loss.detach()/self.args.gradient_accumulation_steps

    def _get_train_sampler(self):
        """
        Provides author sampler.
        """
        train_sampler = ShuffledAuthorSampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size
        )
        return train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use the author-level sampler from dp_transformers.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )