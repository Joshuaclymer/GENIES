from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import PreTrainedModel, Trainer
import fire
import torch.nn as nn
from api.train import train_with_trainer
import torch
from torch.nn import CrossEntropyLoss
import fire
from typing import List, Optional
from api.model import Model
from api.data_classes import MCDataCollator, Distribution, MCDataset
from api.evaluate import compute_metrics

class MCTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_predictions(model, batch, return_outputs=False):
        scores = []
        for input in batch["example_inputs"]:
            if not model.training:
                with torch.no_grad():
                    output = model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])
            else:
                output = model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])
            logits = output.logits

            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input["labels"][:, 1:].contiguous()

            # Flatten the tokens
            shift_logits = shift_logits.reshape(-1, logits.shape[-1])
            shift_labels = shift_labels.reshape(-1)

            # Cross-entropy loss
            loss_fct = CrossEntropyLoss(reduction="none")  # use "none" to get loss per item
            losses = loss_fct(shift_logits, shift_labels)
            avg_log_probs = -losses.view(logits.size(0), -1).mean(dim=1)
            scores.append(avg_log_probs)
        scores = torch.stack(scores, dim=0)

        # Unflatten avg log probs
        response_labels = batch["response_labels"]
        response_labels = response_labels.to(model.device).to(dtype=scores.dtype)

        if model.training:
            response_labels.requires_grad = True
        response_probabilities = torch.softmax(scores, dim=1).to(model.device)
        if return_outputs:
            return response_probabilities, output
        return response_probabilities

    def compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            response_probabilities, outputs = MCTrainer.get_predictions(model, inputs, return_outputs = return_outputs)
        else:
            response_probabilities = MCTrainer.get_predictions(model, inputs, return_outputs = return_outputs)
        correct_response_probabilities = (response_probabilities * inputs["response_labels"]).sum(dim=1)
        loss =  - torch.log(correct_response_probabilities.mean())
        if return_outputs:
            return (loss, outputs)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        predictions = MCTrainer.get_predictions(model, inputs, return_outputs = False)
        predictions_discretized = torch.where(predictions > 0.5, 1.0, 0.0)
        labels_one_hot = inputs["response_labels"]
        labels = torch.argmax(labels_one_hot, dim=1)
        correct_response_probabilities = (predictions * inputs["response_labels"]).sum(dim=1)
        loss =  - torch.log(correct_response_probabilities.mean())
        return loss, predictions, labels

def main(
        model_dir : str,
        output_dir : str,
        training_distribution_dir: str,
        test_distribution_dir : Optional[List[str]] = None, 
        max_eval_examples : int = 100,
        max_train_examples : int = None,
        **kwargs
    ) -> Optional[List[float]]: 

    train_dataset = Distribution(training_distribution_dir).training_dataset
    if test_distribution_dir != None:
        eval_dataset = Distribution(test_distribution_dir).test_dataset
    else:
        eval_dataset = None
    model = Model(model_dir)
    return train_with_trainer(
        model = model,
        output_dir = output_dir,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        trainer = MCTrainer,
        data_collator = MCDataCollator(tokenizer=model.tokenizer),
        max_eval_examples = max_eval_examples,
        compute_metrics = compute_metrics,
        max_train_examples = max_train_examples,
        **kwargs
    )

def fire_wrapper(*args, **kwargs):
    main(*args, **kwargs)

if __name__ == "__main__":
    fire.Fire(fire_wrapper)