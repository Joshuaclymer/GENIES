import copy
import pickle
from functools import partial
from typing import List, Optional

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from baukit import TraceDict
from datasets import load_dataset
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
)
from tqdm import tqdm

import api.util as util
import interventions.mms.llama as llama
from api.data_classes import Distribution
from api.model import Model


def post_hoc_calibration(
    val_set_idxs,
    directions,
    separated_head_wise_activations,
    separated_labels,
    top_heads,
):
    scores = []
    labels = []
    sign = 1
    for i in val_set_idxs:

        def get_activations_for_top_heads(all_activations):
            head_activations = []
            for layer, head in top_heads:
                head_activations.append(torch.tensor(all_activations[layer, head]))
            return torch.stack(head_activations, dim=0)

        activations = [
            get_activations_for_top_heads(a) for a in separated_head_wise_activations[i]
        ]
        sign = -sign
        direction_estimate = sign * torch.tensor(activations[1] - activations[0])
        scores.append(
            [
                0.5
                * (
                    1
                    + F.cosine_similarity(direction_estimate, directions, dim=1).mean()
                )
            ]
        )
        if sign == 1:
            labels.append(int(separated_labels[i][1]))
        else:
            labels.append(int(separated_labels[i][0]))

    calibrator = LogisticRegression(penalty="none").fit(scores, labels)
    calibrated_probs = calibrator.predict_proba(scores)[:, 1]
    print("Brier Score before Calibration:", brier_score_loss(labels, scores))
    print("Brier Score after Calibration:", brier_score_loss(labels, calibrated_probs))
    return calibrator


def get_llama_activations_bau(model, prompt, device):
    model.eval()

    HEADS = [
        f"model.layers.{i}.self_attn.head_out"
        for i in range(model.config.num_hidden_layers)
    ]
    MLPS = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS + MLPS) as ret:
            output = model(prompt, output_hidden_states=True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim=0).squeeze()
        head_wise_hidden_states = [
            ret[head].output.squeeze() for head in HEADS
        ]  # TODO: remove [0] for llama
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim=0)
        mlp_wise_hidden_states = [ret[mlp].output.squeeze() for mlp in MLPS]
        mlp_wise_hidden_states = torch.stack(mlp_wise_hidden_states, dim=0)

    return hidden_states, head_wise_hidden_states, mlp_wise_hidden_states


def get_top_heads(
    train_idxs,
    val_idxs,
    separated_activations,
    separated_labels,
    num_layers,
    num_heads,
    seed,
    num_to_intervene,
    use_random_dir=False,
):
    probes, all_head_accs_np = train_probes(
        seed,
        train_idxs,
        val_idxs,
        separated_activations,
        separated_labels,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)

    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads * num_layers))[::-1][
        :num_to_intervene
    ]
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir:
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(
            num_heads * num_layers, num_heads * num_layers, replace=False
        )
        top_heads = [
            flattened_idx_to_layer_head(idx, num_heads)
            for idx in random_idxs[:num_to_intervene]
        ]

    return top_heads, probes


ENGINE_MAP = {
    "llama_7B": "decapoda-research/llama-7b-hf",
    "alpaca_7B": "circulus/alpaca-7b",
    "vicuna_7B": "AlekseyKorshuk/vicuna-7b",
    "llama2_chat_7B": "meta-llama/Llama-2-7b-chat-hf",
}


def get_llama_logits(model, prompt, device):
    model.eval()
    with torch.no_grad():
        prompt = prompt.to(device)
        logits = model(prompt).logits
        logits = logits.detach().cpu()
        return logits


def save_probes(probes, path):
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, "wb") as f:
        pickle.dump(probes, f)


def load_probes(path):
    """loads a list of sklearn lr probes from path"""
    with open(path, "rb") as f:
        probes = pickle.load(f)
    return probes


def run_ce_loss(
    model_key,
    model=None,
    tokenizer=None,
    device="cuda",
    interventions={},
    intervention_fn=None,
    num_samples=100,
):
    # load owt text
    # note this is tokenized with llama tokenizer
    dataset = load_dataset("stas/openwebtext-10k")["train"]
    dataset = dataset.shuffle()
    dataset = dataset.select(range(num_samples))

    # tokenize
    owt = dataset.map(
        lambda x: {
            "input_ids": torch.tensor(
                tokenizer(x["text"], return_tensors="pt")["input_ids"][:, :128]
            )
        }
    )
    owt.set_format(type="torch", columns=["input_ids"])

    # define intervention
    def id(o_projput, layer_name):
        return o_projput

    if interventions == {}:
        layers_to_intervene = []
        intervention_fn = id
    else:
        layers_to_intervene = list(interventions.keys())
        intervention_fn = partial(intervention_fn, start_edit_location=0)

    losses = []
    rand_idxs = np.random.choice(len(owt), num_samples, replace=False).tolist()
    with torch.no_grad():
        for i in tqdm(rand_idxs):
            input_ids = owt[i]["input_ids"][:, :128].to(device)

            with TraceDict(
                model, layers_to_intervene, edit_output=intervention_fn
            ) as ret:
                loss = model(input_ids, labels=input_ids).loss

            losses.append(loss.item())

    return np.mean(losses)


def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads


def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head


def train_probes(
    seed,
    train_set_idxs,
    val_set_idxs,
    separated_head_wise_activations,
    separated_labels,
    num_layers,
    num_heads,
):
    all_head_accs = []
    probes = []

    all_X_train = np.concatenate(
        [separated_head_wise_activations[i] for i in train_set_idxs], axis=0
    )
    all_X_val = np.concatenate(
        [separated_head_wise_activations[i] for i in val_set_idxs], axis=0
    )
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis=0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis=0)

    for layer in tqdm(range(num_layers)):
        for head in range(num_heads):
            X_train = all_X_train[:, layer, head, :]
            X_val = all_X_val[:, layer, head, :]

            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(
                X_train, y_train
            )
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)

    return probes, all_head_accs_np


def get_interventions_dict(
    top_heads,
    probes,
    tuning_activations,
    num_heads,
    use_center_of_mass,
    use_random_dir,
    com_directions,
):
    interventions = {}
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.head_out"] = []
    for layer, head in top_heads:
        if use_center_of_mass:
            direction = com_directions[
                layer_head_to_flattened_idx(layer, head, num_heads)
            ]
        elif use_random_dir:
            direction = np.random.normal(size=(128,))
        else:
            direction = probes[
                layer_head_to_flattened_idx(layer, head, num_heads)
            ].coef_
        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:, layer, head, :]  # batch x 128
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"model.layers.{layer}.self_attn.head_out"].append(
            (head, direction.squeeze(), proj_val_std)
        )
    for layer, head in top_heads:
        interventions[f"model.layers.{layer}.self_attn.head_out"] = sorted(
            interventions[f"model.layers.{layer}.self_attn.head_out"],
            key=lambda x: x[0],
        )
    return interventions


def get_separated_activations(labels, head_wise_activations):
    # separate activations by question
    # dataset=load_dataset('truthful_qa', 'multiple_choice')['validation']
    # actual_labels = []
    # for i in range(len(dataset)):
    # actual_labels.append(dataset[i]['mc2_targets']['labels'])

    # idxs_to_split_at = np.cumsum([len(x) for x in actual_labels])
    idxs_to_split_at = [i for i in range(len(labels)) if i % 2 == 0 and i != 0]

    # labels = list(labels)
    # separated_labels = []
    # for i in range(len(idxs_to_split_at)):
    #     if i == 0:
    #         separated_labels.append(labels[:idxs_to_split_at[i]])
    #     else:
    #         separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    # assert separated_labels == actual_labels

    separated_labels = np.split(labels, idxs_to_split_at)
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, idxs_to_split_at


def get_com_directions(
    num_layers,
    num_heads,
    train_set_idxs,
    val_set_idxs,
    separated_head_wise_activations,
    separated_labels,
):
    com_directions = []

    for layer in range(num_layers):
        for head in range(num_heads):
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate(
                [
                    separated_head_wise_activations[i][:, layer, head, :]
                    for i in usable_idxs
                ],
                axis=0,
            )
            usable_labels = np.concatenate(
                [separated_labels[i] for i in usable_idxs], axis=0
            )
            true_mass_mean = np.mean(
                usable_head_wise_activations[usable_labels == 1], axis=0
            )
            false_mass_mean = np.mean(
                usable_head_wise_activations[usable_labels == 0], axis=0
            )
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)

    return com_directions


def main(
    model_dir: str,
    output_dir: str,
    training_distribution_dir: str,
    training_dataset=None,
    model: Optional[Model] = None,
    max_examples: int = None,
    num_intervention_heads: int = 16,
    val_ratio: float = 0.3,
    seed=42,
    alpha=1,
    do_calibration=True,
    **kwargs,
) -> Optional[List[float]]:
    """
    Specify dataset name as the first command line argument. Current options are
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the
    validation set for the specified dataset on the last token for llama-7B.
    """

    dataset1 = Distribution(training_distribution_dir).training_dataset
    dataset1.convert_to_pairs(one_pair_per_instruction=True)
    dataset1.set_max_examples(max_examples)
    dataset2 = copy.deepcopy(dataset1)

    postfix1 = "\n\n### Does the responder successfully follow the instruction?\nYes"
    postfix2 = "\n\n### Does the responder successfully follow the instruction?\nNo"

    for e in dataset1.examples:
        e["prompt"] = e["prompt"]
        e["responses"] = {r + postfix1: e["responses"][r] for r in e["responses"]}
    for e in dataset2.examples:
        e["responses"] = {r + postfix2: e["responses"][r] for r in e["responses"]}

    device = "cuda"
    if model == None:
        hf_model = llama.LLaMAForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = Model(model_dir, hf_model=hf_model, quantization_config=None)

    labels = []
    for e in dataset1.examples:
        for response in e["responses"]:
            label = int(e["responses"][response])
            labels.append(label)
    labels2 = []
    for e in dataset2.examples:
        for response in e["responses"]:
            label = int(e["responses"][response])
            labels2.append(label)
    assert labels == labels2

    train_idxs = np.arange(len(dataset1.examples))

    # pick a val set using numpy
    train_set_idxs = np.random.choice(
        train_idxs, size=int(len(train_idxs) * (1 - val_ratio)), replace=False
    )
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
    # define number of layers and heads
    num_layers = model.hf_model.config.num_hidden_layers
    num_heads = model.hf_model.config.num_attention_heads

    def get_activations(model, dataset, seed):
        tokenized_prompts = []
        # Tokenize prompts and get labels
        for e in dataset.examples:
            for response in e["responses"]:
                prompt = e["prompt"] + response
                label = int(e["responses"][response])
                tokenized = model.tokenizer(
                    prompt, return_tensors="pt", padding=False, truncation=False
                )["input_ids"]
                tokenized_prompts.append(tokenized)

        # all_layer_wise_activations = []
        all_head_wise_activations = []

        print("Getting activations:::")
        for prompt in tqdm(tokenized_prompts):
            _, head_wise_activations, _ = get_llama_activations_bau(
                model.hf_model, prompt, device
            )
            # all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
            # print(head_wise_activations.shape) # LGTM
            all_head_wise_activations.append(
                head_wise_activations[:, -1, :].cpu().to(dtype=torch.float32).numpy()
            )

        # set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # load activations
        head_wise_activations = all_head_wise_activations
        head_wise_activations = rearrange(
            head_wise_activations, "b l (h d) -> b l h d", h=num_heads
        )

        # tuning dataset: no labels used, just to get std of activations along the direction

        (
            separated_head_wise_activations,
            separated_labels,
            idxs_to_split_at,
        ) = get_separated_activations(labels, head_wise_activations)
        assert all([len(x) == 2 for x in separated_head_wise_activations])
        assert all([len(x) == 2 for x in separated_labels])
        assert len(separated_head_wise_activations) == len(separated_labels)
        # get directions
        print("Getting directions")
        com_directions = get_com_directions(
            num_layers,
            num_heads,
            train_set_idxs,
            val_set_idxs,
            separated_head_wise_activations,
            separated_labels,
        )
        return separated_labels, separated_head_wise_activations, com_directions

    separated_labels, separated_directions1, directions1 = get_activations(
        model, dataset1, seed
    )
    _, separated_directions2, directions2 = get_activations(model, dataset2, seed)
    com_directions = directions1 - alpha * directions2
    separated_head_wise_activations = [
        [a[0] - alpha * b[0], a[1] - alpha * b[1]]
        for a, b in zip(separated_directions1, separated_directions2)
    ]

    print("Getting top heads")
    top_heads, probes = get_top_heads(
        train_set_idxs,
        val_set_idxs,
        separated_head_wise_activations,
        separated_labels,
        num_layers,
        num_heads,
        seed,
        num_intervention_heads,
        False,
    )
    top_heads = [(int(t[0]), int(t[1])) for t in top_heads]
    directions = []
    for layer, head in top_heads:
        directions.append(
            torch.tensor(
                com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
            )
        )
    directions = torch.stack(directions, dim=0)

    config = {"heads": top_heads, "model_dir": model_dir}
    util.save_json(config, output_dir + "/config.json")
    torch.save(directions, output_dir + "/directions.pt")

    # Fit linear regression model for calibration
    if do_calibration:
        calibrator = post_hoc_calibration(
            val_set_idxs,
            directions,
            separated_head_wise_activations,
            separated_labels,
            top_heads,
        )
        with open(f"{output_dir}/calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)

    return directions, config


if __name__ == "__main__":
    fire.Fire(main)
