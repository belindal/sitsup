import torch
from torch import nn
from torch import optim
import shutil

import numpy as np

from transformers import BartConfig, T5Config
from transformers import BartTokenizerFast, T5TokenizerFast
from transformers import BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import AdamW
from transformers.models.bart.modeling_bart import BartEncoder

import pdb
import argparse
import os
from tqdm import tqdm
import copy
import json
import logging
import random
import glob
from data.dataloader import load_data, convert_to_transformer_batches, split_data_by_final_state
from data import DATASET_REGISTRY, CONSISTENCY_FN_REGISTRY
from data.trip_dataloader import TRIPDatasetGPT3, TRIPDataLoaderGPT3
import random
from state_em_utils import get_expected_state


def eval_model(model, dev_dataloader, state_model=None, output_write_file=None, eval_state=False):
    print("Evaluating")
    metrics = {
        "lm_em": [],
        "lm_state_f1": [],
        "lm_state_em": [],
        "lm_state_binaccept_em_gpt3": [],
        "lm_state_binaccept_em_gt": [],
        "lm_state_binaccept_f1_gt": [],
        "n_val": 0,
        # "loss": 0.0,
    }
    if output_write_file is not None:
        wf = open(output_write_file, "w")
    if model is not None: model.eval()
    if state_model is not None: state_model.eval()
    state_losses = []
    lang_losses = []
    with torch.no_grad():
        pbar = tqdm(dev_dataloader)
        max_shape = 0
        for batch in pbar:
            lang_loss = None
            state_loss = None
            batch_info = [{} for _ in batch["contexts"]["input_ids"]]
            # if metrics["n_val"] > 100: break
            if model is not None:
                return_dict = model(**batch["contexts"], labels=batch['labels']['input_ids'], return_dict=True)
                lang_output = model.generate(**batch['contexts'])  #, output_scores=True)
                lang_output_str = tokenizer.batch_decode(lang_output, skip_special_tokens=True)
                for ex_idx, output_str in enumerate(lang_output_str):
                    metrics["lm_em"].append(output_str == tokenizer.decode(batch['labels']['input_ids'][ex_idx], skip_special_tokens=True))
                    batch_info[ex_idx] = {
                        **batch_info[ex_idx],
                        "inputs": tokenizer.decode(batch['contexts']["input_ids"][ex_idx], skip_special_tokens=True),
                        "gt_label": tokenizer.decode(batch['labels']["input_ids"][ex_idx], skip_special_tokens=True),
                        "gen_label": output_str,
                        "id": batch['ids'][ex_idx],
                    }
                lang_loss = return_dict.loss
                lang_losses.append(lang_loss)
            if eval_state and "facts_input" in batch:
                state = batch["facts_input"]['input_ids']
                kwargs = {}
                if args.decoder_condition_on_prev_state:
                    state_labels = batch["decoder_outputs"]['input_ids']
                    max_shape = max(max_shape, state_labels.shape[1])
                    maxlength = 353
                else:
                    maxlength = 128
                    state_labels = batch['facts_input']['input_ids']
                state_loss = state_model(
                    input_ids=batch['contexts']['input_ids'][batch["facts_mask"]],
                    attention_mask=batch['contexts']['attention_mask'][batch["facts_mask"]], labels=state_labels)[0]
                state_losses.append(state_loss)
            
                if output_write_file is not None or random.random() < 1.0:
                    # include everything....
                    all_state_outputs = state_model.generate(
                        input_ids=batch['contexts']['input_ids'], attention_mask=batch['contexts']['attention_mask'],
                        decoder_start_token_id=state_model.config.pad_token_id, max_length=maxlength,
                    )
                    all_state_outputs_str = tokenizer.batch_decode(all_state_outputs, skip_special_tokens=True)
                    for ex_idx, pred_state_str in enumerate(all_state_outputs_str):
                        if batch["facts_mask"][ex_idx]:
                            facts_idx = batch["facts_mask"].nonzero(as_tuple=False).squeeze().tolist().index(ex_idx)
                            gt_state = set(tokenizer.decode(batch["facts_input"]["input_ids"][facts_idx], skip_special_tokens=True).split("[SEP]")[0].strip().split(". "))
                        if args.decoder_condition_on_prev_state:
                            last_pred_state = pred_state_str.split(" | ")[-1]
                            output_state = set(last_pred_state.strip().split(". "))
                            batch_info[ex_idx]["decoder_inputs"] = kwargs.get("decoder_input_ids", None)
                            gt_label = tokenizer.decode(batch['labels']["input_ids"][ex_idx], skip_special_tokens=True)
                            if batch["facts_mask"][ex_idx]:
                                metrics["lm_state_binaccept_em_gpt3"].append((last_pred_state == "Not OK" and "Not OK" in gt_state) or (last_pred_state != "Not OK" and  "Not OK" not in gt_state))
                            metrics["lm_state_binaccept_em_gt"].append((last_pred_state == "Not OK" and gt_label == "Not OK") or (last_pred_state != "Not OK" and gt_label != "Not OK"))
                        else:
                            output_state = set(pred_state_str.split("[SEP]")[0].strip().split(". "))
                        if "inputs" not in batch_info[ex_idx]:
                            batch_info[ex_idx]["inputs"] = tokenizer.decode(batch['contexts']["input_ids"][ex_idx], skip_special_tokens=True)
                        batch_info[ex_idx]["gen_state"] = list(output_state)
                        if batch["facts_mask"][ex_idx]:
                            intersection = output_state.intersection(gt_state)
                            metrics["lm_state_em"].append(gt_state == output_state)
                            if len(intersection) == 0:
                                metrics["lm_state_f1"].append(0)
                            else:
                                precision = len(intersection) / len(output_state)
                                recall = len(intersection) / len(gt_state)
                                metrics["lm_state_f1"].append((2 * precision * recall) / (precision + recall))
                            batch_info[ex_idx]["gt_state"] = list(gt_state)
            metrics["n_val"] += len(batch["contexts"]["input_ids"])
            if output_write_file is not None:
                for ex_info in batch_info:
                    wf.write(json.dumps(ex_info)+"\n")
                    wf.flush()
            pbar.set_description(
                f"Eval lang loss: {sum(lang_losses) / len(lang_losses) if len(lang_losses) > 0 else 0:.4f}, EM: {sum(metrics['lm_em']) / len(metrics['lm_em']) if len(metrics['lm_em']) > 0 else 0:.4f}; " +
                f"state loss: {sum(state_losses) / len(state_losses) if len(state_losses) > 0 else 0:.4f}, F1: {sum(metrics['lm_state_f1']) / len(metrics['lm_state_f1']) if len(metrics['lm_state_f1']) > 0 else 0:.4f}; " + \
                f"state lang EM: {sum(metrics['lm_state_binaccept_em_gpt3']) / len(metrics['lm_state_binaccept_em_gpt3']) if len(metrics['lm_state_binaccept_em_gpt3']) > 0 else 0:.4f} // {sum(metrics['lm_state_binaccept_em_gt']) / len(metrics['lm_state_binaccept_em_gt']) if len(metrics['lm_state_binaccept_em_gt']) > 0 else 0:.4f}"
            )
    print("EVAL results")
    for metric in metrics:
        if metric != "n_val" and len(metrics[metric]) > 0:
            print(metric, sum(metrics[metric]) / len(metrics[metric]))
    return metrics


def eval_checkpoint(
    args, i, ntrain, train_state, model, dev_dataloader, save_dir, best_val_metric,
    state_model=None, state_save_dir=None, eval_state=False,
):
    stdout_message = [f"EPOCH {i}"]
    # avg_val_loss = 0
    print("Evaluating lang model")
    metrics = eval_model(model, dev_dataloader, state_model=state_model, eval_state=eval_state)
    n_val = metrics["n_val"]
    avg_val_metric = sum(metrics[args.metric]) / len(metrics[args.metric])
    stdout_message.append(f"CONSISTENCY: avg val {args.metric} - {avg_val_metric}")

    print("; ".join(stdout_message))
    
    if avg_val_metric > best_val_metric:
        print("NEW BEST MODEL")
        if model:  torch.save(model.state_dict(),savePath)
        if state_model: torch.save(state_model.state_dict(), state_savePath)
        if poststate_model: torch.save(poststate_model.state_dict(), poststate_savePath)
        new_best_metric = True
    else:
        print(f"model val {args.metric} went {'up' if 'loss' in args.metric else 'down'}")
        new_best_metric = False
    return avg_val_metric, new_best_metric


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='t5-base', choices=['bart-base', 'bart-large', 't5-base', 't5-large', 't5-3b'])
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--eval_batchsize', type=int, default=16)
parser.add_argument('--data', type=str, default="TRIP_dataset")
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--probe_encoder', action='store_true', default=False)
parser.add_argument('--decoder_condition_on_prev_state', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--state_save_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_size', type=int, default=-1)
parser.add_argument('--state_data_size', type=int, default=-1)
parser.add_argument('--rescale_aux_loss', type=float, default=1, help="factor to multiply aux loss")
parser.add_argument('--train_state', type=str, default=None, choices=[
    None,
    'LM_aux_relevant_state', 'only_fact_relevant_state', 'lang_to_relevant_state', 'relevant_state_to_lang',
    'LM_aux_gpt3_state', 'only_fact_gpt3_state', 'lang_to_gpt3_state',
])
parser.add_argument('--force_overwrite_checkpoint', action='store_true', default=False, help="force overwrite the checkpoint (even if eval metric is worse than original)")
parser.add_argument('--metric', type=str, default="lm_em", choices=["lm_em", "lm_state_f1", "lm_state_em"])
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--state_em_cycle_period', type=int, default=-1, help="how many epochs to rerun EM (set 0 to never rerun, -1 for no state)")
parser.add_argument('--lmauxstate_load_path', type=str, default=None)
parser.add_argument('--poststate_load_path', type=str, default=None)
args = parser.parse_args()

arch = args.arch
save_dir = args.save_dir
batchsize = args.batchsize
eval_batchsize = args.eval_batchsize
train_state = args.train_state
device = 'cuda'
metric = args.metric

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

# get arch-specific settings and tokenizers
if 'bart' in arch:
    model_class = BartForConditionalGeneration
    config_class = BartConfig
    model_fp = f'facebook/{arch}'
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
elif 't5' in arch:
    model_class = T5ForConditionalGeneration
    config_class = T5Config
    model_fp = arch
    tokenizer = T5TokenizerFast.from_pretrained('t5-base')
else:
    raise NotImplementedError()

# load or create model(s)
all_models = []
if not args.save_path:
    if not args.save_dir:
        save_dir = (f"model_checkpoints_new/{arch}_lr{args.lr}_TRIP{'_'+args.train_state if args.train_state else ''}"+\
                    f"{'dec_state_history' if args.decoder_condition_on_prev_state else ''}"+\
                    f"_{'em'+str(args.state_em_cycle_period) if args.state_em_cycle_period > -1 else 'ft'}_"+\
                    f"{str(args.state_data_size)+'.' if args.state_data_size > -1 else ''}"+\
                    f"{str(args.train_data_size) if args.train_data_size > -1 else ''}"+\
                    f"{'_auxscale'+str(args.rescale_aux_loss) if args.rescale_aux_loss != 1 else ''}"
                    f"{f'_probe' if args.probe_encoder else ''}{'_seed'+str(args.seed)}"+f"/lang_models")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    savePath = os.path.join(save_dir, f"best_{args.metric}.p")
else:
    savePath = args.save_path
if args.lmauxstate_load_path and os.path.exists(args.lmauxstate_load_path):
    lang_load_path = os.path.join(args.lmauxstate_load_path, 'lang_models', f"best_{args.metric}.p")
    if os.path.exists(lang_load_path):
        shutil.copy(lang_load_path, savePath)
if train_state and train_state.startswith("lang_to"):
    # get rid of lang model
    model = None
else:
    load_model = False
    if os.path.exists(savePath):
        model_dict = torch.load(savePath)
        load_model = True
        print("Loading LM model")
    model = model_class.from_pretrained(model_fp)
    if not load_model:
        print("Creating LM model")
    else:
        model_dict = {k.replace('module.', ''): model_dict[k] for k in model_dict}
        model.load_state_dict(model_dict)
    print(f"    model path: {savePath}")
    config = model.config
    all_models.append(model)

state_model = None
state_save_dir = None
if train_state:
    if not args.save_path:
        state_save_dir = os.path.split(save_dir)
        assert state_save_dir[-1] == 'lang_models'
        state_save_dir = os.path.join(state_save_dir[0], 'state_models')
        if not os.path.exists(state_save_dir): os.makedirs(state_save_dir)
        state_savePath = os.path.join(state_save_dir, f'best_{metric}.p')
    else:
        state_savePath = os.path.split(savePath)
        assert state_savePath[0] == 'lang_models'
        state_savePath = os.path.join('state_models', state_savePath[1])
    if args.lmauxstate_load_path and os.path.exists(args.lmauxstate_load_path):
        state_load_path = os.path.join(args.lmauxstate_load_path, 'state_models', f'best_{metric}.p')
        if state_load_path and os.path.exists(state_load_path):
            shutil.copy(state_load_path, state_savePath)
    load_model = False
    if os.path.exists(state_savePath):
        state_model_dict = torch.load(state_savePath)
        load_model = True
        print("Loading state head")
    if not load_model:
        state_model = model_class.from_pretrained(model_fp)
    else:
        config = config_class.from_pretrained(model_fp)
        state_model = model_class(config)
        state_model.load_state_dict(state_model_dict)
    if model is not None:
        if hasattr(state_model, 'encoder'): state_model.encoder = model.get_encoder()
        else: state_model.model.encoder = model.get_encoder()
    state_model.to(device)
    print(f"    state model path: {state_savePath}")
    all_models.append(state_model)
poststate_model = None
poststate_save_dir = None
if args.state_em_cycle_period != -1:
    if not args.save_path:
        poststate_save_dir = os.path.split(save_dir)
        assert poststate_save_dir[-1] == 'lang_models'
        poststate_save_dir = os.path.join(poststate_save_dir[0], 'poststate_models')
        if not os.path.exists(poststate_save_dir): os.makedirs(poststate_save_dir)
        poststate_savePath = os.path.join(poststate_save_dir, f'best_{metric}.p')
    else:
        poststate_savePath = os.path.split(savePath)
        assert poststate_savePath[0] == 'lang_models'
        poststate_savePath = os.path.join('poststate_models', poststate_savePath[1])
    if args.poststate_load_path and os.path.exists(args.poststate_load_path):
        shutil.copy(args.poststate_load_path, poststate_savePath)
    load_model = False
    if os.path.exists(poststate_savePath):
        poststate_model_dict = torch.load(poststate_savePath)
        load_model = True
        print("Loading state head")
    if not load_model:
        poststate_model = model_class.from_pretrained(model_fp)
    else:
        config = config_class.from_pretrained(model_fp)
        poststate_model = model_class(config)
        poststate_model.load_state_dict(poststate_model_dict)
    poststate_model.to(device)
    print(f"    post-state model path: {poststate_savePath}")    
    all_models.append(poststate_model)
if args.probe_encoder:
    encoder = model.get_encoder()
    for p in encoder.parameters():
        p.requires_grad = False
if model is not None:
    model.to(device)

# load optimizer
all_parameters = []
for m in all_models: all_parameters += [p for p in m.parameters() if p.requires_grad]
all_parameters = list(set([p for p in m.parameters() if p.requires_grad]))
optimizer = AdamW(all_parameters, lr=args.lr)

# load data
dev_dataset = TRIPDatasetGPT3(args.data, "dev", train_state=train_state, sentence_wise=True, data_size=(200 if not args.eval_only else -1), state_data_size=args.state_data_size)
dev_dataloader = TRIPDataLoaderGPT3(
    dev_dataset, tokenizer, eval_batchsize, device,
    decoder_inputs=("state_history" if args.decoder_condition_on_prev_state else None),
    input_type="state" if train_state and "state_to" in train_state else "lang")
# get state (even if we won't use it) so that we have the correct ordering
dataset = TRIPDatasetGPT3(args.data, "train", train_state=train_state, sentence_wise=True, data_size=args.train_data_size, state_data_size=args.state_data_size, seed=args.seed)
train_dataloader = TRIPDataLoaderGPT3(
    dataset, tokenizer, batchsize, device,
    decoder_inputs=("state_history" if args.decoder_condition_on_prev_state else None),
    input_type="state" if train_state and "state_to" in train_state else "lang")
print(f"Loaded data: {len(dataset)} train examples, {len(dev_dataset)} dev examples")

output_json_fn = None
state_output_json_fn = None
if args.eval_only:
    output_json_fn = f"{savePath[:-2]}.jsonl"
    print(f"Saving predictions to {output_json_fn}")

# initial eval
print("Initial eval")
avg_val_metric = 0
n_val = None
if args.eval_only or load_model:
    metrics = eval_model(
        model, dev_dataloader,
        state_model=state_model
        output_write_file=output_json_fn,
        eval_state=args.metric.startswith("lm_state") or args.eval_only,
    )
    avg_val_metric = sum(metrics[args.metric]) / len(metrics[args.metric])
    print(f"VAL METRICS: {args.metric} - {avg_val_metric}")
    n_val = metrics["n_val"]
best_loss_epoch = -1
best_metric_epoch = -1
best_val_metric = avg_val_metric

if args.eval_only:
    exit()


# training loop
print("Start training")
num_updates = 0
best_metric_update = 0
best_loss_update = 0
for i in range(args.epochs):
    if (i - best_metric_epoch > args.patience) and (i - best_loss_epoch > args.patience): break
    ntrain = 0
    if model: model.train()
    if state_model: state_model.train()
    train_losses = []

    if args.state_em_cycle_period != -1 and i % args.state_em_cycle_period == 0:
        """
        e-step: get expected states
        """
        expect_tgt_states = get_expected_state(
            state_model, poststate_model, "trip", dataset, train_dataloader, tokenizer, 16, train_state, nnegs=0, num_samples=5,
            do_eval=False, max_gt_grounded_states=args.state_data_size, post_state_setting="post_state_model", debug=args.debug,
        )
        train_dataloader.dataset.set_expected_states(expect_tgt_states)
    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for j, batch in pbar:
        optimizer.zero_grad()
        bs = batch['contexts']['input_ids'].size(0)  # can be different from `batchsize`
        train_loss = 0.0
        if model is not None:
            return_dict = model(
                **batch['contexts'], labels=batch['labels']['input_ids'], return_dict=True,
            )
            lang_loss = return_dict.loss
            train_loss += lang_loss
        if state_model is not None and "facts_input" in batch:
            if args.decoder_condition_on_prev_state:
                state_labels = batch["decoder_outputs"]['input_ids']
                state_labels_mask = batch["facts_mask"]
            elif args.state_em_cycle_period != -1:
                state_labels = batch['facts_expected_input']['input_ids']
                state_labels_mask = torch.ones(batch['facts_expected_input']['input_ids'].shape[0]).bool().to(batch['facts_expected_input']['input_ids'].device)
            else:
                state_labels = batch['facts_input']['input_ids']
                state_labels_mask = batch["facts_mask"]
            state_loss = state_model(
                input_ids=batch['contexts']['input_ids'][state_labels_mask],
                attention_mask=batch['contexts']['attention_mask'][state_labels_mask],
                labels=state_labels,
            )[0]
            train_loss += args.rescale_aux_loss * state_loss
        if poststate_model is not None and "facts_expected" in batch:
            post_state_loss = poststate_model(
                input_ids=state_labels,
                labels=batch['labels']['input_ids'][state_labels_mask],
            )[0]
            train_loss += post_state_loss
        if train_loss > 0:
            train_losses.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            num_updates += 1
            pbar.set_description(f"Train loss: {sum(train_losses) / len(train_losses)}")
    if args.train_data_size <= 100 and i % 10 != 9: continue
    avg_val_metric, new_best_metric = eval_checkpoint(
        args, i, ntrain, train_state, model, dev_dataloader, save_dir, best_val_metric,
        state_model=state_model,
        state_save_dir=state_save_dir,
        eval_state=args.metric.startswith("lm_state"),
    )
    if new_best_metric:
        best_val_metric = avg_val_metric
        best_metric_epoch = i
