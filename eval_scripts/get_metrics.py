import json
import pdb
import os
# from torchtext.data.metrics import bleu_score
import glob
from data.dataloader import load_data as load_data_TW
from data.cooking_dataloader import CookingDataset
from data.trip_dataloader import TRIPDataset
from data.openpi_dataloader import OpenPIDataset
from transformers import BartTokenizerFast, T5Tokenizer
import itertools 
from tqdm import tqdm
from metrics.tw_metrics import METRICS_TO_EVAL_FN as TW_METRICS_TO_EVAL_FN, get_loss
from metrics.recipes_metrics import METRICS_TO_EVAL_FN as RECIPES_METRICS_TO_EVAL_FN
import numpy as np
import argparse
from scipy.stats import sem 
from numpy import std
import torch

VARYING_TOTAL_LANG_EXS = ['lang_only', 'state_only', 'lang2state', 'lang_state_concat', 'lang_state', 'entity_only', 'lang2state_probe', 'lang2state_probe_control']
VARYING_TOTAL_ALIGNED_EXS = ['lang_state_ft', 'lang_state_em']

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, choices=['bart', 'bart-large', 'bart-base'], default='bart-base', help='which architecture to evaluate')
parser.add_argument('--domain', type=str, choices=['textworld', 'recipes', 'openpi', 'trip'], required=True, help='which domain to evaluate')
parser.add_argument('--state_key', type=str, default=None, help='use something other than the default state_key for the domain')
parser.add_argument('--exp', type=str, choices=VARYING_TOTAL_LANG_EXS+VARYING_TOTAL_ALIGNED_EXS, required=True, help='which experiment to evaluate')
args = parser.parse_args()


def get_actions_feedbacks(generations, gt_continuations, correct_feedback, gt_utterance, ctxt):
    has_action = gt_utterance.startswith('> ')
    has_feedback = gt_utterance.strip(' | ').count(' | ') > 0 or not gt_utterance.startswith('> ')

    if has_action:
        actions = [gen.split(' | ')[0] for gen in generations]
    else:
        actions = ctxt[ctxt.rfind('> '):].split(' | ')[0]
    gt_actions = ["> " + s.split(' | ')[0] for s in gt_continuations]
    if has_feedback:
        if has_action:
            feedbacks = [' | '.join(gen.split(' | ')[1:]) for gen in generations]
            gt_feedbacks = correct_feedback
        else:
            feedbacks = generations
            gt_feedbacks = [gt_utterance]
        feedbacks = [fb.split(' | ')[0] if '-=' in fb else fb for fb in feedbacks]
        gt_feedbacks = [gtfb.split(' | ')[0] if '-=' in gtfb else gtfb for gtfb in gt_feedbacks]
    else:
        feedbacks, gt_feedbacks = None, None
    return has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks

n_samples = 5
ctxt_to_valid_continuations = {}
model_fp = 'facebook/bart-base'
tokenizer = BartTokenizerFast.from_pretrained(model_fp)
state_key = args.state_key
if args.domain == 'textworld':
    dataset = "training_traces_tw-treasure_hunter"
    suffix = "_sep_action_response"
    eval_type = ""
    if not state_key: state_key = "curr_state_belief"
    fact_sep_token = ' [SEP] '
    # eval_type = "contrastive"
    outputs_fn = f"best_lm_multi_bleu{n_samples}_samples"
    probe_data_size = 32966
    # get gt continuations
    dev_data = load_data_TW(
        "tw_games/training_traces_tw-treasure_hunter/dev", tokenizer, 1024, max_data_size=10000, pred_action_and_response_joint=False,
    )
    dev_id_order = []
    # dev_data["filenames"]
    for i, prev_ctxts in enumerate(dev_data['contexts']):
        game_id = dev_data['filenames'][i].split('_')[0]
        ctxt_to_valid_continuations[(game_id, prev_ctxts)] = dev_data['final_state'][i]['valid_actions']
        dev_id_order.extend([dev_data["filenames"][i] for _ in range(n_samples)])
    
    # define metrics
    METRICS = ['CONSISTENCY', 'MULTIREF_BLEU', 'CONCAT_ROUGE', 'F1', 'EM']  #'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'F1']
    # eval_type = "_evalcontrastive" #"_evalcontrastive" / "contrastive" 
    # METRICS = ['EM']
    # outputs_fn = "best_lm_em1_samples" #"best_lm_em1_samples" / "best_lm_multi_bleu1_samples"
    METRICS_TO_EVAL_FN = TW_METRICS_TO_EVAL_FN
    model_name = f"pre_{args.arch}"
elif args.domain == 'recipes':
    dataset = "recipes"
    suffix = ""
    eval_type = "_evalcontrastive"
    if not state_key: state_key = "curr_state_full"
    fact_sep_token = ', '
    outputs_fn = "best_lm_em1_samples"
    probe_data_size = 817797
    # get gt continuations
    dev_data = CookingDataset(
        "cooking_dataset/recipes", tokenizer, "dev", 1024, max_data_size=100000,
    )
    for i, dev_entry in enumerate(dev_data):
        recipe_id = dev_entry['filenames']
        prev_ctxts = tokenizer.decode(tokenizer.encode(dev_entry['context']), skip_special_tokens=True)
        ctxt_to_valid_continuations[(recipe_id, prev_ctxts)] = [tokenizer.decode(tokenizer.encode(dev_entry['next_instrs']), skip_special_tokens=True)]
    METRICS = ['EM']
    METRICS_TO_EVAL_FN = RECIPES_METRICS_TO_EVAL_FN
elif args.domain == 'openpi':
    dataset = "openPI_data"
    suffix = ""
    eval_type = "_evalcontrastive"
    if not state_key: state_key = "curr_state"
    fact_sep_token = ', '
    outputs_fn = "best_lm_em1_samples"
    probe_data_size = 3194
    # get gt continuations
    dev_data = OpenPIDataset(
        "openPI_data", tokenizer, "dev", 1024, max_data_size=100000,
    )
    METRICS = ['EM']
    METRICS_TO_EVAL_FN = RECIPES_METRICS_TO_EVAL_FN
elif args.domain == 'trip':
    dataset = "TRIP"
    model_name = args.arch
    suffix = ""
    eval_type = "_ft" #"_evalcontrastive"
    if not state_key: state_key = "relevant_state"
    outputs_fn = "best_lm_em"
    fact_sep_token = '. '
    probe_data_size = 3238
    # get gt continuations
    dev_data = TRIPDataset(
        "TRIP_dataset", tokenizer, "dev", 1024, max_data_size=100000,
    )
    # for i, dev_entry in enumerate(dev_data):
    #     ex_id = dev_entry['example_id']
    #     prev_ctxts = tokenizer.decode(tokenizer.encode(dev_entry['context']), skip_special_tokens=True)
    #     ctxt_to_valid_continuations[(ex_id, prev_ctxts)] = [tokenizer.decode(tokenizer.encode(dev_entry['next_instrs']), skip_special_tokens=True)]
    METRICS = ['EM', 'pos_difference']
    METRICS_TO_EVAL_FN = RECIPES_METRICS_TO_EVAL_FN


if args.exp == 'lang_only':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}{eval_type}_{train_data_size}_seed{seed}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_recipes_evalcontrastive_{lang_data_size}_seed{seed}/lang_models/best_lm_bleu1_samples.jsonl"
    #     for lang_data_size in [1000] for seed in [0,1,2,3,42]
    # ]
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_TRIP_dataset_evalcontrastive_{lang_data_size}_seed{seed}/lang_models/best_lm_em1_samples.jsonl"
    #     for lang_data_size in [100,1000,4000] for seed in [0,1,2,3,42]
    # ]
elif args.exp == 'state_only':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_only_fact_{state_key}{eval_type}_{train_data_size}_seed{seed}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'entity_only':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_controlinput{eval_type}_{train_data_size}_seed{seed}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'lang_state_concat':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_concat_state_{state_key}{eval_type}_{train_data_size}_seed{seed}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'lang2state':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_lang_to_{state_key}_{lang_data_size}_seed{seed}{suffix}/lang_models/best_lm_bleu1_samples.jsonl"
        for lang_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,5,6,7,42]
    ]
    METRICS = ['F1']
elif args.exp == 'lang2state_probe':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_lang_to_{state_key}_{lang_data_size}_seed{seed}_probe{probe_data_size}.{lang_data_size}{suffix}/lang_models/best_lm_bleu1_samples.jsonl"
        for lang_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
    METRICS = ['F1']
elif args.exp == 'lang2state_probe_control':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_lang_to_{state_key}_{lang_data_size}_seed{seed}_probe_control{suffix}/lang_models/best_lm_bleu1_samples.jsonl"
        for lang_data_size in [100,1000,2000,3200,4000,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
    METRICS = ['F1']
elif args.exp == 'lang_state':
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}{eval_type}_{lang_data_size}_seed{seed}{suffix}/lang_models/{outputs_fn}.jsonl"
    #     for train_data_size in [1.0] for lang_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,5,6,7,42]
    # ]
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}{eval_type}_{lang_data_size}_seed{seed}_ft{int(train_data_size*lang_data_size)}.{lang_data_size}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [0.0] for lang_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_recipes_LM_aux_curr_state_evalcontrastive_{lang_data_size}_seed{seed}_ftinf/lang_models/best_lm_bleu1_samples.jsonl"
    #     for lang_data_size in [1000] for seed in [0,1,2,3,42]
    # ]
elif args.exp == 'lang_state_ft':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}{eval_type}_{lang_data_size}_seed{seed}_ft{int(train_data_size*lang_data_size)}.{lang_data_size}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [0,0.01,0.05,0.1,0.5,1.0] for lang_data_size in [1000] for seed in [0,1,2,3,4,5,6,7,42]
        # for train_data_size in [0.1] for lang_data_size in [1000] for seed in [0]
    ]
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_recipes_LM_aux_curr_state_evalcontrastive_{lang_data_size}_seed{seed}_ft{int(train_data_size*lang_data_size)}.{lang_data_size}/lang_models/best_lm_bleu1_samples.jsonl"
    #     for train_data_size in [0.0,0.01,0.05,0.1,0.5] for lang_data_size in [1000] for seed in [0,1,2,3,4,5,6,7,42]
    # ]
elif args.exp == 'lang_state_em':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}{eval_type}_{lang_data_size}_seed{seed}_em{int(train_data_size*lang_data_size)}.{lang_data_size}{suffix}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [0,0.01,0.05,0.1,0.5,1.0] for lang_data_size in [1000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
    # lm_outputs = [
    #     f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}{eval_type}_{lang_data_size}_seed{seed}_em{int(train_data_size*lang_data_size)}.{lang_data_size}{suffix}/lang_models/best_lm_multi_bleu_ensemble5_1.05_samples.jsonl"
    #     for train_data_size in [0,0.01,0.05,0.1,0.5,1.0] for lang_data_size in [1000] for seed in [0,1,2,3,4,5,6,7,42]
    # ]

print(lm_outputs)
lm_outputs = [fp for fp in lm_outputs if os.path.exists(fp)]
detail_level = 0

Xs = []
for lm_output_fn in lm_outputs:
    if args.exp in VARYING_TOTAL_LANG_EXS:
        X = int(lm_output_fn.split('_seed')[0].split('_')[-1])
    else:
        assert args.exp in VARYING_TOTAL_ALIGNED_EXS
        X = int(lm_output_fn.split('/')[1].split('.')[0].split('_ft')[-1].split('_em')[-1])
    if len(Xs) == 0:
        Xs.append(X)
    else:
        assert X >= max(Xs)
        if X > max(Xs):
            Xs.append(X)
print(Xs)

all_metrics = {
    metric: [] for metric in METRICS
}
if 'F1' in METRICS:
    all_metrics['PRE'] = []
    all_metrics['REC'] = []
all_metrics['LOSS'] = []
fn_noseed_to_idx = {}
for lm_output in lm_outputs:
    invalid = False
    # with open(lm_output) as f:
    #     # game_id_to_curr_file = {}
    #     for line in f:
    #         try: line = json.loads(line)
    #         except:
    #             invalid = True
    #             break
    # if invalid: continue

    print(lm_output)

    # get seed (for aggregating metrics by seed)
    seed = lm_output[lm_output.find('_seed'):][len('_seed'):]
    seed = int(seed[:min(seed.find('_'), seed.find('/'))])
    fn_noseed = lm_output.replace(f'_seed{seed}', '')
    if fn_noseed not in fn_noseed_to_idx:
        for metric in all_metrics:
            all_metrics[metric].append([])
        fn_noseed_to_idx[fn_noseed] = len(all_metrics[metric]) - 1

    # get predicted generations
    ctxt_to_generations = {}
    ctxt_to_lines = {}
    # story_id_to_metrics = {}
    if args.domain == 'textworld':
        ctxt_to_gt_utterance = {}
        ctxt_to_correct_feedback = {}
    # get metrics
    correct_feedback = {}
    valid_but_incorrect = {}
    curr_run_metrics = {
        metric: [0,0] for metric in METRICS
    }
    if 'F1' in METRICS:
        curr_run_metrics['PRE'] = [0,0]
        curr_run_metrics['REC'] = [0,0]
    prf1s = []

    # get lm perplexities
    curr_run_metrics['LOSS'] = get_loss(os.path.join(os.path.split(os.path.split(lm_output)[0])[0], "train.log"))
    if args.exp in ["lang2state", "lang2state_probe", "lang2state_probe_control"]:
        with open(lm_output) as f:
            ln = 0
            for line in f:
                line = json.loads(line)
                assert 'F1' in METRICS
                real_facts = set(line['gt_utt'].split(fact_sep_token))
                pred_facts = set(line['gen_utt'].split(fact_sep_token))
                correct_pred_facts = real_facts.intersection(pred_facts)
                precision = len(correct_pred_facts) / len(pred_facts)
                recall = len(correct_pred_facts) / len(real_facts)
                if (precision + recall) > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0
                curr_run_metrics['PRE'][0] += precision
                curr_run_metrics['PRE'][1] += 1
                curr_run_metrics['REC'][0] += recall
                curr_run_metrics['REC'][1] += 1
                curr_run_metrics['F1'][0] += f1
                curr_run_metrics['F1'][1] += 1
    elif 'contrastive' in eval_type and args.domain != 'textworld':
        with open(lm_output) as f:
            ln = 0
            for line in f:
                line = json.loads(line)
                assert 'EM' in METRICS
                if 'correct?' in line:
                    curr_run_metrics['EM'][0] += line['correct?']
                else:
                    curr_run_metrics['EM'][0] += (line['gt_utt'] == line['gen_utt'])
                curr_run_metrics['EM'][1] += 1
                if 'pos_difference' in METRICS:
                    # negative log likelihoods
                    curr_run_metrics['pos_difference'][0] += min(line['scores']['neg']) - line['scores']['pos']
                    curr_run_metrics['pos_difference'][1] += 1
    else:
        with open(lm_output) as f:
            ln = 0
            ctxt_to_story_ids = {}
            for line in f:
                line = json.loads(line)
                id = line.get('game_id', None)
                # assert dev_id_order[ln].split("_")[0] == id
                prev_ctxt = line['prev_context']
                # ctxt_to_story_ids[(id, prev_ctxt)] = dev_id_order[ln]
                if (id, prev_ctxt) not in ctxt_to_generations:
                    ctxt_to_generations[(id, prev_ctxt)] = []
                    ctxt_to_lines[(id, prev_ctxt)] = []
                    if args.domain == 'textworld':
                        ctxt_to_correct_feedback[(id, prev_ctxt)] = []
                        ctxt_to_gt_utterance[(id, prev_ctxt)] = line['gt_utt']
                ctxt_to_generations[(id, prev_ctxt)].append(line['gen_utt'])
                ctxt_to_lines[(id, prev_ctxt)].append(line)
                if args.domain == 'textworld':
                    ctxt_to_correct_feedback[(id, prev_ctxt)].append(line['correct_feedback'])
                    try:
                        if not line['gt_utt'].startswith('>'):
                            assert ctxt_to_gt_utterance[(id, prev_ctxt)] == line['gt_utt']
                    except:
                        import pdb; pdb.set_trace()
                ln += 1
            # assert ln == len(dev_id_order)

        syntax_errors = 0
        num_invalid = 0
        num_total = 0
        num_actually_invalid = 0
        num_wrong_feedback = 0
        num_actually_wrong_feedback = 0
        skipped_actions = 0
        all_generations = {'action': {}, 'feedback': {}}
        for (id, ctxt) in ctxt_to_generations:
            # get generations/all gt continuations/selected gt coninuation
            generations = ctxt_to_generations[(id, ctxt)]
            num_total += len(generations)

            has_feedback = False
            if args.domain == 'textworld':
                gt_continuations = ctxt_to_valid_continuations[(id, ctxt)]
                correct_feedback = ctxt_to_correct_feedback[(id, ctxt)]
                gt_utterance = ctxt_to_gt_utterance[(id, ctxt)]
                # get action/feedback
                (has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks) = get_actions_feedbacks(
                    generations, gt_continuations, correct_feedback, gt_utterance, ctxt,
                )
                # if has_feedback:
                #     continue
                # breakpoint()
                # detect syntax errors
                for gen in generations:
                    syntax_errors += (has_action and not gen.startswith('> ')) or (
                        has_feedback and gen.strip(' | ').count(' | ') == 0 and gen.startswith('> ')
                    ) or (gt_utterance.startswith('> ') and not gen.startswith('> '))
                metric_arguments = [has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, ctxt_to_lines[(id, ctxt)]]
            elif args.domain == 'recipes':
                gt_continuations = ctxt_to_valid_continuations[(id, ctxt)]
                metric_arguments = [generations, gt_continuations, ctxt_to_lines[(id, ctxt)]]

            # scores = {metric: METRICS_TO_EVAL_FN[metric](has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks) for metric in METRICS}
            for metric in METRICS:
                if 'F1' in metric:
                    if not has_feedback:
                        p,r,f1 = METRICS_TO_EVAL_FN[metric](*metric_arguments)
                        curr_run_metrics['PRE'][0] += p
                        curr_run_metrics['PRE'][1] += 1
                        curr_run_metrics['REC'][0] += r
                        curr_run_metrics['REC'][1] += 1
                        curr_run_metrics['F1'][0] += f1
                        curr_run_metrics['F1'][1] += 1
                        prf1s.append([p,r,f1])
                else:
                    curr_run_metrics[metric][0] += METRICS_TO_EVAL_FN[metric](*metric_arguments)
                    curr_run_metrics[metric][1] += 1
                # story_id = ctxt_to_story_ids[(id, ctxt)]
                # if story_id not in story_id_to_metrics:
                #     breakpoint()
                #     story_id_to_metrics[story_id] = {}
                # if metric not in story_id_to_metrics[story_id]:
                #     story_id_to_metrics[story_id][metric] = []
                # story_id_to_metrics[story_id][metric].append(curr_run_metrics[metric][0])

    if len(curr_run_metrics['LOSS']) > 0:
        print(f"Best loss: {min(curr_run_metrics['LOSS'])}")
        all_metrics['LOSS'][fn_noseed_to_idx[fn_noseed]].append(min(curr_run_metrics['LOSS']))
    
    prf1s = np.array(prf1s)
    metrics_to_print = METRICS.copy()
    if 'F1' in METRICS:
        metrics_to_print += ['PRE', 'REC']
    for metric in metrics_to_print:
        if curr_run_metrics[metric][1] > 0:
            curr_run_metrics[metric] = curr_run_metrics[metric][0] / curr_run_metrics[metric][1]
        else:
            curr_run_metrics[metric] = 0
        print(f"{metric}: {curr_run_metrics[metric]}")
        all_metrics[metric][fn_noseed_to_idx[fn_noseed]].append(curr_run_metrics[metric] * 100)

    if detail_level > 0:
        print(f"Syntax: {syntax_errors} / {num_total} = {(syntax_errors / num_total) * 100}%")
        # print(f"Invalids: {num_invalid} / {num_total} = {num_invalid / num_total}; {1 - num_invalid / num_total}")
        print(f"Invalids: {num_actually_invalid} / {num_total} = {(num_actually_invalid / num_total) * 100}%")
        if detail_level > 1:
            sorted_feedback_freq = sorted([k for k in correct_feedback.keys()], key=lambda x: len(correct_feedback[x]), reverse=True)
            print("\t"+"\n\t".join([f"{x} {len(correct_feedback[x]) / num_total * 100}" for x in sorted_feedback_freq]))
        print(f"Wrong feedback: {num_actually_wrong_feedback} / {num_total} = {(num_actually_wrong_feedback / num_total) * 100}%")
        if detail_level > 1:
            sorted_feedback_freq = sorted([k for k in valid_but_incorrect.keys()], key=lambda x: len(valid_but_incorrect[x]), reverse=True)
            print("\t"+"\n\t".join([f"{x} {len(valid_but_incorrect[x]) / num_total * 100}" for x in sorted_feedback_freq]))


print(" ==== ")
print(f"{{\"Xs\": {Xs}}}")
for metric in all_metrics:
    print(json.dumps({metric: all_metrics[metric]}))
    print(json.dumps({
        metric: [f"{torch.tensor(all_metrics[metric][x_idx]).mean():.2f} +/- {sem(all_metrics[metric][x_idx]):.2f}" for x_idx, x_metric in enumerate(all_metrics[metric])]
    }))
    # storywise_metrics = []
    # for story in story_id_to_metrics:
    #     breakpoint()
    #     storywise_metrics.append(all(story_id_to_metrics[story][metric]))
    # print(f"storywise {metric}: {torch.tensor(storywise_metrics).float().mean()*100:.2f} +/- {sem(storywise_metrics)*100:.2f}")
    # for story in story_id_to_metrics:
    #     print(f"{metric} {}")