import pandas as pd
import datasets
from datasets import Dataset
import torch
from transformers import (
    TrainerCallback, TrainerState, TrainerControl, logging,
    PreTrainedTokenizer, training_args
)
import opacus
from opacus.accountants import RDPAccountant
from prv_accountant import Accountant as PRVAccountant
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from accelerate.optimizer import AcceleratedOptimizer

from trl import DataCollatorForCompletionOnlyLM

logger = logging.get_logger(__name__)

class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """
    def __init__(
        self,
        noise_multiplier: float,
        target_delta: float,
        sampling_probability: float,
        rdp_accountant: RDPAccountant,
        prv_accountant: PRVAccountant,
        max_epsilon: float = float('inf')
    ) -> None:

        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.sampling_probability = sampling_probability
        self.rdp_accountant = rdp_accountant
        self.prv_accountant = prv_accountant

        self.max_epsilon = max_epsilon
        self.on_substep_end_was_called = False
        self.compute_rdp_epsilon = lambda: self.rdp_accountant.get_epsilon(self.target_delta)
        self.compute_prv_epsilon = lambda s: self.prv_accountant.compute_epsilon(s)[2]

    def on_substep_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        if isinstance(optimizer, AcceleratedOptimizer):
            dp_optimizer = optimizer.optimizer
        else:
            dp_optimizer = optimizer
        dp_optimizer.signal_skip_step(do_skip=True)
        dp_optimizer.step()
        dp_optimizer.zero_grad()

        self.on_substep_end_was_called = True

    def on_step_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if not (
            args.gradient_accumulation_steps <= 1 or
            self.on_substep_end_was_called
        ):
            raise RuntimeError(
                "Gradient accumulation was specified but `on_substep_end` wasn't called. "
                "Make sure you're using a recent version of transformers (>=4.10.0) "
                "which has an appropriate callback in the trainer."
            )

        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        optimizer.zero_grad()  # Opacus is bothered that HF does not call .zero_grad() on the optimizer

        self.rdp_accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sampling_probability)

    def on_save(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(state, control)

    def on_evaluate(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._check_max_epsilon_exceeded(state, control)

    def _check_max_epsilon_exceeded(self, state: TrainerState, control: TrainerControl) -> TrainerControl:
        eps_rdp = self.compute_rdp_epsilon()
        eps_prv = self.compute_prv_epsilon(state.global_step)
        if eps_rdp > self.max_epsilon or eps_prv > self.max_epsilon:
            logger.error("Max epsilon exceeded. Stopping training...")
            control.should_training_stop = True
        return control


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForCompletionOnlyLM):
    def __init__(self, response_template, tokenizer: PreTrainedTokenizer):
        super().__init__(response_template, tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here
    """
    @contextmanager
    def no_sync(self):
        yield


def create_author_mapping(dataset: Dataset, author: str) -> Sequence[Sequence[int]]:
    """
    Creates a mapping from authors to samples in a dataset.
    """
    with dataset.formatted_as(type="pandas"):
        authors = pd.DataFrame(data={"author": dataset[author]})
        author_mapping = [g.index.values for _, g in authors.groupby("author")]
    return author_mapping
