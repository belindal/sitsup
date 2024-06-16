import torch
from torch import nn
from torch import optim

import numpy as np

import textworld
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
from utils import DEVICE, consistencyCheck, get_tw_consistency
from data.dataloader import load_data, convert_to_transformer_batches, split_data_by_final_state
from data import DATASET_REGISTRY, CONSISTENCY_FN_REGISTRY
from data.trip_dataloader import att_to_num_classes
from train_lm_utils import eval_model
from train_state_utils import eval_state_model
from state_em_utils import get_expected_state
# from models.models import ContrastiveClassifierHead, JointClassifierHead,
# LinearEncoderDecoderInterace, TRIPEntityStateModel


def eval_checkpoint(
    args, i, ntrain, train_state, model, dev_dataloader, save_dir,
    best_val_loss, best_val_metric, eval_type=None,
    state_model=None, state_save_dir=None, post_state_model=None, poststate_save_dir=None,
):
    print(f"n_train {ntrain}")

    stdout_message = [f"EPOCH {i}"]
    avg_val_loss = 0
    if args.data_type == 'textworld':
        dev_dataloader = convert_to_transformer_batches(
            args, dev_dataset, tokenizer, eval_batchsize, train_state=train_state,
            # append_facts_to_input=(train_state.startswith('input') if train_state else train_state),
            nnegs=(6 if eval_type == 'contrastive' else 0), npos=(float('inf') if (args.metric == "lm_multi_bleu") and eval_type != 'contrastive' else 1),
        )
    if args.metric == "lm_loss" or args.metric == "lm_state_loss" or args.metric == "lm_bleu" or args.metric == "lm_multi_bleu" or args.metric == "lm_em" or args.metric == "lm_state_f1":
        print("Evaluating lang model")
        results = eval_model(
            args, model, dev_dataloader, tokenizer, eval_type=eval_type, train_state=train_state,
            get_bleu=(args.metric == "lm_bleu"),
            get_multi_bleu=(args.metric == "lm_multi_bleu"), num_samples=args.num_samples,
            dataset=dev_dataset, state_model=state_model,
        )
        n_val = results[0]
        avg_val_loss += results[1]
        other_metrics = results[6]
        if args.metric == "lm_loss": avg_val_metric = results[1]
        if args.metric == "lm_bleu": avg_val_metric = results[3]
        if args.metric == "lm_multi_bleu": avg_val_metric = results[4]
        if args.metric == "lm_em": avg_val_metric = results[5]
        if args.metric == "lm_state_f1": avg_val_metric = other_metrics["state_f1"]
        stdout_message.append(f"CONSISTENCY: avg val loss - {results[1]}, {args.metric} - {avg_val_metric}")
    if args.metric == "state_loss" or args.metric == "lm_state_loss":
        print("Evaluating state model")
        results = eval_state_model(
            args, state_model, dev_dataset, tokenizer, eval_batchsize, state_keys_to_get=[train_state], post_state_setting=post_state_setting)
        n_val = results[0]
        avg_val_loss += results[1]
        stdout_message.append(f"STATE: avg val loss - {results[1]}, EM - {results[2]}, F1 - {results[3]}")
        if post_state_model:
            print("Evaluating post-state model")
            results = eval_post_state_model(
                args, post_state_model, dev_dataset, tokenizer, eval_batchsize,
                state_keys_to_get=[train_state],
            )
            avg_val_loss += results[1]
            stdout_message.append(f"POST STATE: avg val loss - {results[1]}")

    stdout_message.append(f"OVERALL: avg val loss - {avg_val_loss}")
    print("; ".join(stdout_message))
    if "loss" in args.metric: avg_val_metric = avg_val_loss

    # save checkpoints
    if not args.save_path:
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch{i}.p"))
        if state_model: torch.save(state_model.state_dict(), os.path.join(state_save_dir, f"epoch{i}.p"))
        if post_state_model: torch.save(post_state_model.state_dict(), os.path.join(poststate_save_dir, f"epoch{i}.p"))
    new_best_loss = avg_val_loss < best_val_loss
    if ("loss" in args.metric and avg_val_loss < best_val_loss) or (avg_val_metric > best_val_metric):
        print("NEW BEST MODEL")
        model.epoch = i
        torch.save(model.state_dict(),savePath)
        if state_model: torch.save(state_model.state_dict(), state_savePath)
        if objs_can_interact_model: torch.save(objs_can_interact_model.state_dict(), objs_savePath)
        if post_state_model: torch.save(post_state_model.state_dict(), poststate_savePath)
        new_best_metric = True
    else:
        print(f"model val {args.metric} went {'up' if 'loss' in args.metric else 'down'}")
        new_best_metric = False
    return avg_val_loss, avg_val_metric, new_best_loss, new_best_metric


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='bart-base', choices=['bart-base', 'bart-large', 't5-base', 't5-large', 't5-3b'])
parser.add_argument('--override_num_layers', type=int, default=None)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--eval_batchsize', type=int, default=16)
parser.add_argument('--control_input', action='store_true', default=False, help="only condition on initial state with entities")
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--data_type', type=str, required=True, choices=['textworld', 'recipes', 'openpi', 'trip'])
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--do_state_em', action='store_true', default=False)
parser.add_argument('--ensemble_weight', type=float, default=0.5)
parser.add_argument('--ensemble_samples', type=int, default=-1)
parser.add_argument('--eval_interval', type=int, default=-1, help="evaluate every this number of updates, set to -1 for eval only at end of each epochs")
parser.add_argument('--eval_state_em', action='store_true', default=False)
parser.add_argument('--encoder_layer', type=int, default=-1, help="which layer of encoder to decode from")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--eval_only', action='store_true', default=False)
parser.add_argument('--eval_type', type=str, default="decoder") # contrastive, contrastive_synth
# parser.add_argument('--freeze_decoder', action='store_true', default=False)
parser.add_argument('--probe_encoder', action='store_true', default=False)
parser.add_argument('--gamefile', type=str, required=False)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--max_gt_grounded_states', type=int, default=-1, help="if training with state, max amount of gold state-supervision")
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--no_pretrain', action='store_true', default=False)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--state_save_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--train_data_size', type=int, default=4000)
parser.add_argument('--train_state', type=str, default=None, choices=[
    None,
    'only_fact_full', 'only_fact_added_full', 'only_fact_curr_state_full', 'only_fact_same_room_rephrased_full', 'only_fact_added_can_interact_full',
    'only_fact_belief', 'only_fact_added_belief', 'only_fact_curr_state_belief', 'only_fact_same_room_rephrased_belief', 'only_fact_added_can_interact_belief',
    'input_belief', 'input_full',
    'LM_aux_full', 'LM_aux_added_full', 'LM_aux_curr_state_full', 'LM_aux_same_room_rephrased_full', 'LM_aux_added_can_interact_full',
    'LM_aux_belief', 'LM_aux_added_belief', 'LM_aux_curr_state_belief', 'LM_aux_same_room_rephrased_belief', 'LM_aux_added_can_interact_belief',
    'lang_to_curr_state_belief', 'lang_to_curr_state_full',
    'LM_sep_curr_state_belief', 'LM_sep_curr_state_full',
    'contrastive_aux', 'interleave_in_ctxt',
    'LM_aux_state_change',
    'concat_fact_curr_state_belief', 'concat_fact_curr_state_belief',
    # openPI
    'LM_aux_curr_state', 'only_fact_curr_state', 'lang_to_curr_state',
    # cooking
    'only_fact_changed_curr_state', 'lang_to_changed_curr_state',
    'only_fact_relevant_state_full', 'LM_aux_changed_curr_state', 'LM_aux_changed_events', 'LM_aux_changed_entities',
    'concat_fact_relevant_state', 'lang_to_relevant_state',
    # TRIP
    'LM_aux_relevant_state', 'only_fact_relevant_state', 'lang_to_relevant_state',
    'LM_aux_preconditions', 'LM_aux_effects', 'LM_aux_preconditions_effects',
    'lang_to_preconditions', 'lang_to_effects','lang_to_preconditions_effects',
    'only_fact_preconditions', 'only_fact_effects', 'only_fact_preconditions_effects', #'LM_aux_attributes', 'only_fact_attributes',
    'only_fact_relevant_state_entity', 'LM_aux_relevant_state_entity', 'LM_aux_preconditions_entity', 'LM_aux_effects_entity',
    'LM_aux_gpt3_state', 'only_fact_gpt3_state', 'lang_to_gpt3_state',
])
parser.add_argument('--aux_obj_in_room', action='store_true', default=False)
parser.add_argument('--objs_save_path', type=str, default='')
parser.add_argument('--gen_split', action='store_true', default=False)
parser.add_argument('--force_overwrite_checkpoint', action='store_true', default=False, help="force overwrite the checkpoint (even if eval metric is worse than original)")
parser.add_argument('--metric', type=str, default="lm_loss", choices=["lm_loss", "state_loss", "lm_state_loss", "lm_bleu", "lm_multi_bleu", "lm_em", "lm_state_f1"])
parser.add_argument('--pred_action_and_response_sep', action='store_true', default=False)
# parser.add_argument('--use_post_state_model', action='store_true', default=False)
parser.add_argument('--post_state_setting', type=str, choices=[None, 'post_state_model', 'early_fusion', 'late_fusion'])
parser.add_argument('--em_cycle_period', type=int, default=1, help="how many epochs to rerun EM (set 0 to never rerun)")
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--local_files_only', action='store_true', default=False, help="use pretrained checkpoints saved in local directories")
args = parser.parse_args()

arch = args.arch
save_dir = args.save_dir
pretrain = not args.no_pretrain
batchsize = args.batchsize
eval_batchsize = args.eval_batchsize
eval_interval = args.eval_interval
ensemble_samples = args.ensemble_samples
ensemble_weight = args.ensemble_weight
max_seq_len = args.max_seq_len
eval_type = args.eval_type
generalization_split = args.gen_split
train_data_size = args.train_data_size
aux_obj_in_room = args.aux_obj_in_room  # add auxiliary loss for whether obj is in room
force_overwrite_checkpoint = args.force_overwrite_checkpoint
inform7_game = None
em_cycle_period = args.em_cycle_period
if em_cycle_period == 0: em_cycle_period = float("inf")
metric = args.metric
if args.max_gt_grounded_states == -1 and args.train_state: max_gt_grounded_states = float("inf")
else: max_gt_grounded_states = args.max_gt_grounded_states
pred_action_and_response_sep = args.pred_action_and_response_sep
post_state_setting = args.post_state_setting
if args.data_type == 'textworld':
    # maybe inexact (grammars between worlds might be different(?)) but more efficient
    for fn in glob.glob(os.path.join(args.gamefile, 'train/*.ulx')):
        env = textworld.start(fn)
        game_state = env.reset()
        game_kb = game_state['game'].kb.inform7_predicates
        inform7_game = env._inform7
        break

# seed everything
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

train_state = None
if args.train_state:
    if 'only_fact' in args.train_state:
        train_state = ['only_fact', f'{args.train_state.split("only_fact_")[-1]}_facts']
    elif 'concat_fact' in args.train_state:
        train_state = ['concat_fact', f'{args.train_state.split("concat_fact_")[-1]}_facts']
    elif 'lang_to' in args.train_state:
        train_state = ['lang_to', f'{args.train_state.split("lang_to_")[-1]}_facts']
    elif 'LM_aux' in args.train_state:
        train_state = ['LM_aux', f'{args.train_state.split("LM_aux_")[-1]}_facts']
    elif 'LM_sep' in args.train_state:
        train_state = ['LM_sep', f'{args.train_state.split("LM_sep_")[-1]}_facts']
if args.probe_encoder:
    assert train_state and train_state[0] == 'lang_to'

# get arch-specific settings and tokenizers
if 'bart' in arch:
    model_class = BartForConditionalGeneration
    config_class = BartConfig
    model_fp = f'facebook/{arch}'
    tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base', local_files_only=args.local_files_only)
elif 't5' in arch:
    model_class = T5ForConditionalGeneration
    config_class = T5Config
    model_fp = arch
    tokenizer = T5TokenizerFast.from_pretrained('t5-base', local_files_only=args.local_files_only)
else:
    raise NotImplementedError()

# load or create model(s)
load_model = False
if not args.save_path:
    if not args.save_dir:
        save_dir = (f"model_checkpoints/{'pre_' if pretrain else 'nonpre_'}{arch}_{f'{args.override_num_layers if args.override_num_layers else args.encoder_layer}layer_' if args.override_num_layers or args.encoder_layer != -1 else ''}lr{args.lr}_{args.data.split('/')[-1]}"+\
                    f"{'_'+args.train_state if args.train_state else ''}{'_controlinput' if args.control_input else ''}"+\
                    f"{'_aux_objs' if aux_obj_in_room else ''}{'_'+post_state_setting if post_state_setting else ''}{'_eval'+eval_type if eval_type is not 'decoder' else ''}{'_gensplit' if generalization_split else ''}"+\
                    f"{'_'+str(train_data_size)}{'_seed'+str(args.seed)}"+\
                    f"{f'_probe' if args.probe_encoder else ''}{f'_em' if args.do_state_em else ''}{f'_ft' if max_gt_grounded_states > 0 else ''}{f'{max_gt_grounded_states}' if max_gt_grounded_states > 0 else ''}"+\
                    f"{'_sep_action_response' if args.pred_action_and_response_sep else ''}/"+\
                    f"lang_models")
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    savePath = os.path.join(save_dir, f"best_{args.metric}.p")
else:
    savePath = args.save_path
if os.path.exists(savePath):
    try:
        model_dict = torch.load(savePath)
    except: breakpoint()
    load_model = True
    print("Loading LM model")
if not load_model: print("Creating LM model")
if not load_model and pretrain:
    model = model_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
else:
    config = config_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
    if args.override_num_layers is not None:
        if 'bart' in arch:
            setattr(config, 'num_hidden_layers', args.override_num_layers)
            setattr(config, 'encoder_layers', args.override_num_layers)
            setattr(config, 'decoder_layers', args.override_num_layers)
        elif 't5' in arch:
            setattr(config, 'num_layers', args.override_num_layers)
            setattr(config, 'num_decoder_layers', args.override_num_layers)
    model = model_class(config)
if args.encoder_layer != -1:
    # TODO INCOMPATIBLE IF PRETRAINED???
    model.model.encoder.layers = model.model.encoder.layers[:args.encoder_layer]
if load_model:
    # convert to 
    model_dict = {k.replace('module.', ''): model_dict[k] for k in model_dict}
    model.load_state_dict(model_dict)
print(f"    model path: {savePath}")
config = model.config

all_models = [model]
state_model = None
objs_can_interact_model = None
post_state_model = None  # maps from state -> valid transcript continuations
state_save_dir = None
poststate_save_dir = None
if train_state and 'aux' in train_state[0]:
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
    load_model = False
    if os.path.exists(state_savePath):
        state_model_dict = torch.load(state_savePath)
        load_model = True
        print("Loading state head")
    else:
        print("Creating state head")
    if train_state and train_state[0].startswith('LM_'):
        if not load_model and pretrain:
            state_model = model_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
        else:
            config = config_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
            state_model = model_class(config)
            if load_model:
                state_model_dict = {k.replace('module.', ''): state_model_dict[k] for k in state_model_dict}
                state_model.load_state_dict(state_model_dict)
        if '_aux' in train_state[0]:
            if hasattr(state_model, 'encoder'): state_model.encoder = model.get_encoder()
            else: state_model.model.encoder = model.get_encoder()
        else:
            # no parameter sharing
            assert '_sep' in train_state[0]
        if torch.cuda.device_count() > 1: state_model = torch.nn.DataParallel(state_model)
        state_model.to('cuda')
    print(f"    state model path: {state_savePath}")
    all_models.append(state_model)
if post_state_setting == 'post_state_model':
    if not args.save_path:
        poststate_save_dir = os.path.split(save_dir)
        assert poststate_save_dir[-1] == 'lang_models'
        poststate_save_dir = os.path.join(poststate_save_dir[0], 'post_state_models')
        if not os.path.exists(poststate_save_dir): os.makedirs(poststate_save_dir)
        poststate_savePath = os.path.join(poststate_save_dir, f'best_{metric}.p')
    else:
        poststate_savePath = os.path.split(savePath)
        assert poststate_savePath[0] == 'lang_models'
        poststate_savePath = os.path.join('post_state_models', poststate_savePath[1])
    load_model = False
    if os.path.exists(poststate_savePath):
        post_state_model_dict = torch.load(poststate_savePath)
        load_model = True
        print("Loading post state head")
    else:
        print("Creating post state head")
    if not load_model and pretrain:
        post_state_model = model_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
    else:
        config = config_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
        post_state_model = model_class(config)
        if load_model:
            post_state_model_dict = {k.replace('module.', ''): post_state_model_dict[k] for k in post_state_model_dict}
            post_state_model.load_state_dict(post_state_model_dict)
    post_state_model.to(DEVICE)
    print(f"    post-state model path: {poststate_savePath}")
    all_models.append(post_state_model)
if aux_obj_in_room:
    load_model = False
    if os.path.exists(args.objs_save_path):
        objs_savePath = args.objs_save_path
        objs_can_interact_model_dict = torch.load(objs_savePath)
        load_model = True
        print("Loading can-interact-objects head")
    else:
        print("Creating can-interact-objects head")
    if not load_model and pretrain:
        objs_can_interact_model = model_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
    else:
        config = config_class.from_pretrained(model_fp, local_files_only=args.local_files_only)
        objs_can_interact_model = model_class(config)
        if load_model: objs_can_interact_model.load_state_dict(objs_can_interact_model_dict)
    objs_savePath = os.path.split(savePath)
    assert objs_savePath[0] == 'lang_models'
    objs_savePath = os.path.join('obj_can_interact_models', objs_savePath[1])
    if hasattr(objs_can_interact_model, 'encoder'): objs_can_interact_model.encoder = model.get_encoder()
    else: objs_can_interact_model.model.encoder = model.get_encoder()
    if torch.cuda.device_count() > 1: objs_can_interact_model = torch.nn.DataParallel(objs_can_interact_model)
    objs_can_interact_model.to('cuda')
    print(f"    can-interact-objects model path: {objs_savePath}")
    all_models.append(objs_can_interact_model)
if args.probe_encoder:
    encoder = model.get_encoder()
    for p in encoder.parameters():
        p.requires_grad = False
    # TODO randomly initialize decoder???
if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
model.to('cuda')

# load optimizer
all_parameters = []
for m in all_models: all_parameters += [p for p in m.parameters() if p.requires_grad]
all_parameters = list(set(all_parameters))
optimizer = AdamW(all_parameters, lr=args.lr)

# load data
max_data_size = [float("inf") if args.eval_only else 500, train_data_size]
if args.data_type == 'textworld':
    dev_dataset = load_data(
        os.path.join(args.data, 'dev'), tokenizer, max_seq_len, max_data_size=max_data_size[0],
        inform7_game=inform7_game,
        pred_action_and_response_joint=(not args.pred_action_and_response_sep),
        control_input=args.control_input,
    )
    dataset = load_data(
        os.path.join(args.data, 'train'), tokenizer, max_seq_len, max_data_size=max_data_size[1],
        inform7_game=inform7_game,
        pred_action_and_response_joint=(not args.pred_action_and_response_sep), randseed=args.seed,
        control_input=args.control_input,
    )
    train_dataloader = convert_to_transformer_batches(
        args, dataset, tokenizer, batchsize, train_state=train_state,
        nnegs=0, include_feedback=True, max_gt_grounded_states=max_gt_grounded_states,
    )
    dev_dataloader = convert_to_transformer_batches(
        args, dev_dataset, tokenizer, eval_batchsize, train_state=train_state,
        nnegs=(6 if 'contrastive' in eval_type else 0), npos=(float('inf') if (args.metric == "lm_multi_bleu") and 'contrastive' not in eval_type else 1),
    )
    print(f"Loaded data: {len(dataset['contexts'])} train examples, {len(dev_dataset['contexts'])} dev examples")
elif args.data_type in DATASET_REGISTRY:
    dataset_class = DATASET_REGISTRY[args.data_type]['dataset']
    dataloader_class = DATASET_REGISTRY[args.data_type]['dataloader']
    dev_dataset = dataset_class(
        args.data, tokenizer, 'dev', max_seq_len, max_data_size=max_data_size[0], contrastive=(args.eval_type == 'contrastive'), train_state=train_state, control_input=args.control_input,
    )
    dataset = dataset_class(
        args.data, tokenizer, 'train', max_seq_len, max_data_size=max_data_size[1],
        max_gt_grounded_states=max_gt_grounded_states, randseed=args.seed, contrastive=False, train_state=train_state, control_input=args.control_input,
    )
    train_dataloader = dataloader_class(dataset, tokenizer, batchsize, train_state=train_state, contrastive=False)
    dev_dataloader = dataloader_class(dev_dataset, tokenizer, eval_batchsize, train_state=train_state, contrastive=eval_type)
    print(f"Loaded data: {len(dataset)} train examples, {len(dev_dataset)} dev examples")
else:
    raise NotImplementedError()

all_cand_outputs = all_cand_output_encodings = None
output_json_fn = None
state_output_json_fn = None
if args.eval_only:
    output_json_fn = f"{savePath[:-2]+(f'_ensemble{ensemble_samples}_{ensemble_weight}' if ensemble_samples > -1 else '')+f'{args.num_samples}_samples.jsonl'}"
    print(f"Saving predictions to {output_json_fn}")
    # if type(model) is ContrastiveClassifierHead:
    #     all_cand_outputs, all_cand_output_encodings = get_all_outputs(dev_dataset, model=model, tokenizer=tokenizer)
    if state_model is not None:
        state_output_json_fn = os.path.join("state_model_outputs", f"{state_savePath[:-2]+f'{args.num_samples}_samples.jsonl'}")
        print(f"Saving state predictions to {state_output_json_fn}")

# initial eval
print("Initial eval")
avg_val_loss = 0
avg_val_metric = 0
n_val = None
# """
if not args.debug:
    if args.metric == "lm_loss" or args.metric == "lm_state_loss" or args.metric == "lm_bleu" or args.metric == "lm_multi_bleu" or args.metric == "lm_em" or args.metric == "lm_state_f1":
        if args.data_type == 'textworld':
            dev_dataloader = convert_to_transformer_batches(
                args, dev_dataset, tokenizer, eval_batchsize, train_state=train_state,
                # append_facts_to_input=(train_state.startswith('input') if train_state else train_state),
                nnegs=(6 if 'contrastive' in eval_type else 0), npos=(float('inf') if (args.metric == "lm_multi_bleu") and 'contrastive' not in eval_type else 1),
            )
        if not args.eval_only or (train_state and train_state[0] == 'lang_to'):
            consistency_fn = None
        elif args.data_type == 'textworld':
            consistency_fn = get_tw_consistency
        else:
            consistency_fn = CONSISTENCY_FN_REGISTRY[args.data_type]
        results = eval_model(
            args, model, dev_dataloader, tokenizer, eval_type=eval_type, get_consistency=consistency_fn,
            get_bleu=(args.metric == "lm_bleu"), get_multi_bleu=(args.metric == "lm_multi_bleu"), output_json_fn=output_json_fn, num_samples=args.num_samples,
            state_model=state_model, post_state_model=post_state_model, train_state=train_state,
            all_cand_outputs=all_cand_outputs, all_cand_output_encodings=all_cand_output_encodings,
            objs_can_interact_model=objs_can_interact_model, dataset=dev_dataset, post_state_setting=post_state_setting,
            ensemble_samples=ensemble_samples, ensemble_weight=ensemble_weight,
        )
        n_val = results[0]
        avg_val_loss += results[1]
        p_consistent = results[2]
        other_metrics = results[6]
        if args.metric == "lm_loss" or args.metric == "lm_state_loss":
            avg_val_metric += avg_val_loss
        if args.metric == "lm_bleu":
            avg_val_metric += results[3]
        if args.metric == "lm_multi_bleu":
            avg_val_metric += results[4]
        if args.metric == "lm_em":
            avg_val_metric += results[5]
        if args.metric == "lm_state_f1":
            avg_val_metric += other_metrics["state_f1"]
        print(f"CONSISTENCY: loss - {results[1]}, overall - {p_consistent[0]}, %invalid - {p_consistent[1]}, %wrongfeedback - {p_consistent[2]}, {args.metric} - {avg_val_metric}")
    if args.metric == "state_loss" or args.metric == "lm_state_loss":
        results = eval_state_model(
            args, state_model, dev_dataset, tokenizer, eval_batchsize,
            state_keys_to_get=[train_state], output_json_fn=state_output_json_fn,
            post_state_setting=post_state_setting,
        )
        n_val = results[0]
        avg_val_loss += results[1]
        print(f"STATE: loss - {results[1]}, EM - {results[2]}, F1 - {results[3]}")
    if n_val is None: assert False
else:
    print("debug mode")
# """
print(f"avg val loss: {avg_val_loss}")
best_loss_epoch = -1
best_metric_epoch = -1
if force_overwrite_checkpoint:
    best_val_loss = float("inf")
    best_val_metric = 0
else:
    best_val_loss = avg_val_loss
    best_val_metric = avg_val_metric

if args.eval_only:
    exit()


# training loop
print("Start training")
num_updates = 0
best_metric_update = 0
best_loss_update = 0
for i in range(args.epochs):
    # TODO adjust behavior??
    if (i - best_metric_epoch > args.patience) and (i - best_loss_epoch > args.patience): break
    ntrain = 0
    model.train()
    if state_model: state_model.train()
    lang_train_losses = []
    expect_tgt_states = None
    if args.do_state_em and i % em_cycle_period == 0 and not args.debug:
        """
        e-step: get expected states
        """
        expect_tgt_states = get_expected_state(
            state_model, post_state_model, args.data_type, dataset, train_dataloader, tokenizer, 16, train_state, nnegs=0, num_samples=5,
            debug=args.debug, do_eval=args.eval_state_em, max_gt_grounded_states=max_gt_grounded_states, post_state_setting=post_state_setting, args=args,
        )
        if args.data_type != 'textworld':
            train_dataloader.dataset.set_expected_states(expect_tgt_states)
    if args.data_type == 'textworld':
        train_dataloader = convert_to_transformer_batches(
            args, dataset, tokenizer, batchsize, train_state=train_state, #append_facts_to_input=(train_state.startswith('input') if train_state else train_state),
            include_feedback=True, expected_states=expect_tgt_states, max_gt_grounded_states=max_gt_grounded_states,
        )

    state_key = getattr(train_dataloader, 'state_key', train_state[-1] if train_state else None)
    for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entities) in enumerate(train_dataloader):
        if eval_interval > -1 and (num_updates - best_metric_update > args.patience * eval_interval) and (num_updates - best_loss_update > args.patience * eval_interval): break
        if inputs is None: continue
        optimizer.zero_grad()
        bs = inputs['input_ids'].size(0)  # can be different from `batchsize`
        if train_state:
            if args.do_state_em:
                state_to_use = tgt_state[state_key+'_expected']
            else:
                state_to_use = tgt_state[state_key]
            if 'entity' in state_key:
                use_gt_state_mask = torch.tensor([ent is not None for ent in entities])
            else:
                use_gt_state_mask = state_to_use['attention_mask'].sum(1) > 2
            if train_state[0] == "only_fact":
                inputs = tgt_state[state_key+'_input']
            elif train_state[0] == "concat_fact":
                inputs = tgt_state[state_key+'_concat_text']
            elif train_state[0] == "lang_to":
                lang_tgts = state_to_use
        if 'all_cands_input_ids' in lang_tgts:
            # get only valid_actions
            labels = lang_tgts['all_cands_input_ids'][lang_tgts['labels'].bool()]
            n_repeats = lang_tgts['labels'].sum(1).long()
            input_ids = inputs['input_ids'].repeat_interleave(n_repeats,dim=0)
            attn_mask = inputs['attention_mask'].repeat_interleave(n_repeats,dim=0)
            return_dict = model(
                input_ids=input_ids, attention_mask=attn_mask, labels=labels, return_dict=True,
            )
            ntrain += n_repeats.sum()
        else:
            return_dict = model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['input_ids'], return_dict=True,
            )
        lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
        if train_state and (use_gt_state_mask).any():
            # nonempty, excluding bos/eos
            if state_model:
                encoder_outputs = (encoder_hidden[use_gt_state_mask],)
                if 'entity' in state_key:
                    state = {att: state_to_use[att][use_gt_state_mask] for att in state_to_use}
                else:
                    state = state_to_use['input_ids'][use_gt_state_mask]
                    state_attn_mask = state_to_use['attention_mask'][use_gt_state_mask]
                if post_state_setting == 'early_fusion':
                    state_input_ids = inputs['full_input_ids'][use_gt_state_mask]
                    state_attention_mask = inputs['full_attention_mask'][use_gt_state_mask]
                    state_loss = state_model(input_ids=state_input_ids, attention_mask=state_attention_mask, labels=state)[0]
                else:
                    if 'entity' in state_key:
                        encoder_outputs = train_dataloader.select_vectors_corresponding_to_entity(encoder_outputs, inputs, entities)
                        # feed corresponding to `entity`
                        state_loss = state_model(encoder_outputs=encoder_outputs, labels=state)['loss']
                    else:
                        state_loss = state_model(input_ids=None, encoder_outputs=encoder_outputs, labels=state)[0]
                if post_state_setting == 'late_fusion':
                    post_encoder_output = state_model.get_encoder()(
                        input_ids=inputs['post_input_ids'][use_gt_state_mask], attention_mask=inputs['post_attention_mask'][use_gt_state_mask]).last_hidden_state
                    import pdb; pdb.set_trace()
                lang_loss += state_loss
            if objs_can_interact_model:
                objs_can_interact = tgt_state['objs']['input_ids_can_interact']
                objs_can_interact = objs_can_interact[use_gt_state_mask]
                encoder_outputs = (encoder_hidden,[use_gt_state_mask])
                objs_can_interact_loss = objs_can_interact_model(input_ids=None, encoder_outputs=encoder_outputs, labels=objs_can_interact)[0]
                lang_loss += objs_can_interact_loss
            if post_state_model:
                state_inputs = tgt_state[train_state[-1]+'_concat_text']['input_ids']
                state_attn_mask = tgt_state[train_state[-1]+'_concat_text']['attention_mask']
                correspond_txt = lang_tgts['input_ids'][use_gt_state_mask]
                state_inputs = state_inputs[use_gt_state_mask]
                state_attn_mask = state_attn_mask[use_gt_state_mask]
                # guess next gen from state
                post_loss = post_state_model(input_ids=state_inputs, attention_mask=state_attn_mask, labels=correspond_txt, return_dict=True).loss
                lang_loss += post_loss
        lang_loss = lang_loss.mean()
        # encoder_outputs = (encoder_hidden,)
        lang_train_losses.append(lang_loss.item())
        try: lang_loss.backward()
        except: import pdb; pdb.set_trace()
        optimizer.step()
        num_updates += 1
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, loss: {lang_loss.item()}", flush=True)
        if eval_interval > -1 and j%eval_interval == eval_interval - 1:
            avg_val_loss, avg_val_metric, new_best_loss, new_best_metric = eval_checkpoint(
                args, i, ntrain, train_state, model, dev_dataloader, save_dir, best_val_loss, best_val_metric,
                eval_type=eval_type,
                state_model=state_model, state_save_dir=state_save_dir,
                post_state_model=post_state_model, poststate_save_dir=poststate_save_dir,
            )
            if new_best_metric:
                best_val_metric = avg_val_metric
                best_metric_epoch = i
                best_metric_update = num_updates
            if new_best_loss:
                best_val_loss = avg_val_loss
                best_loss_epoch = i
                best_loss_update = num_updates
    avg_val_loss, avg_val_metric, new_best_loss, new_best_metric = eval_checkpoint(
        args, i, ntrain, train_state, model, dev_dataloader, save_dir, best_val_loss, best_val_metric,
        eval_type=eval_type,
        state_model=state_model, state_save_dir=state_save_dir,
        post_state_model=post_state_model, poststate_save_dir=poststate_save_dir,
    )
    if new_best_metric:
        best_val_metric = avg_val_metric
        best_metric_epoch = i
        # best_update = num_updates
    if new_best_loss:
        best_val_loss = avg_val_loss
        best_loss_epoch = i
