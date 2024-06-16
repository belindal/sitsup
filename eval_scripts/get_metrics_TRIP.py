
import json
import argparse
import os
from scipy.stats import sem 
from numpy import std
import torch


VARYING_TOTAL_LANG_EXS = ['lang_only', 'LM_aux_state_full', 'state_to_lang']
VARYING_TOTAL_ALIGNED_EXS = ['LM_aux_state_partial', 'lang_state_em']


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, choices=['bart', 'bart-large', 'bart-base'], default='bart-base', help='which architecture to evaluate')
# parser.add_argument('--state_key', type=str, default=None, help='use something other than the default state_key for the domain')
parser.add_argument('--exp', type=str, choices=VARYING_TOTAL_LANG_EXS+VARYING_TOTAL_ALIGNED_EXS, required=True, help='which experiment to evaluate')
args = parser.parse_args()

dataset = "TRIP"
model_name = args.arch
state_key = "gpt3_state"
outputs_fn = "best_lm_em"
fact_sep_token = '. '
if args.exp == 'lang_only':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_ft_{train_data_size}_seed{seed}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'state_to_lang':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_{state_key}_to_lang_ft_{train_data_size}_seed{seed}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'LM_aux_state_full':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}_ft_{train_data_size}_seed{seed}/lang_models/{outputs_fn}.jsonl"
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'LM_aux_state_partial':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}_ft_{int(state_data_prop*train_data_size)}.{train_data_size}_seed{seed}/lang_models/{outputs_fn}.jsonl"
        for state_data_prop in [0.1,0.5]
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]
elif args.exp == 'lang_state_em':
    lm_outputs = [
        f"model_checkpoints_new/{model_name}_lr1e-05_{dataset}_LM_aux_{state_key}_em5_{int(state_data_prop*train_data_size)}.{train_data_size}_seed{seed}/lang_models/{outputs_fn}.jsonl"
        for state_data_prop in [0.01,0.05,0.1,0.5]
        for train_data_size in [100,1000,2000,3200,5000,10000,100000] for seed in [0,1,2,3,4,5,6,7,42]
    ]

lm_outputs = [fp for fp in lm_outputs if os.path.exists(fp)]

Xs = []
for lm_output_fn in lm_outputs:
    # if args.exp in VARYING_TOTAL_LANG_EXS:
    #     X = int(lm_output_fn.split('_seed')[0].split('_')[-1])
    # else:
    #     assert args.exp in VARYING_TOTAL_ALIGNED_EXS
    # 
    X = int(lm_output_fn.split('/')[1].split('_ft_')[-1].split('_em5_')[-1].split('_')[0].split('.')[0])
    if len(Xs) == 0:
        Xs.append(X)
    else:
        assert X >= max(Xs)
        if X > max(Xs):
            Xs.append(X)
print(Xs)

all_metrics = {
    "EM": [],
}
metrics_to_print = ["EM"]
fn_noseed_to_idx = {}
storywise_metrics = {
    "EM": []
}
for lm_output in lm_outputs:
    print(lm_output)

    # get seed (for aggregating metrics by seed)
    seed = lm_output[lm_output.find('_seed'):][len('_seed'):]
    seed = int(seed[:min(seed.find('_'), seed.find('/'))])
    fn_noseed = lm_output.replace(f'_seed{seed}', '')
    if fn_noseed not in fn_noseed_to_idx:
        for metric in all_metrics:
            all_metrics[metric].append([])
            storywise_metrics[metric].append([])
        fn_noseed_to_idx[fn_noseed] = len(all_metrics[metric]) - 1

    # get predicted generations
    ctxt_to_generations = {}
    ctxt_to_lines = {}
    curr_run_metrics = {
        metric: [0,0] for metric in all_metrics
    }
    story_id_to_metrics = {}
    with open(lm_output) as f:
        for line in f:
            line = json.loads(line)
            curr_run_metrics["EM"][0] += line["gt_label"] == line["gen_label"]
            curr_run_metrics["EM"][1] += 1
            story_id, sent_idx = line["id"].split("_")
            if story_id not in story_id_to_metrics:
                story_id_to_metrics[story_id] = {metric: {} for metric in all_metrics}
            sent_idx = int(sent_idx)
            # assert sent_idx not in story_id_to_metrics[story_id]["EM"]
            story_id_to_metrics[story_id]["EM"][sent_idx] = line["gt_label"] == line["gen_label"]

    for metric in metrics_to_print:
        if curr_run_metrics[metric][1] > 0:
            curr_run_metrics[metric] = curr_run_metrics[metric][0] / curr_run_metrics[metric][1]
        else:
            curr_run_metrics[metric] = 0
        print(f"{metric}: {curr_run_metrics[metric]}")
        all_metrics[metric][fn_noseed_to_idx[fn_noseed]].append(curr_run_metrics[metric] * 100)
        all_story_accs = []
        for story in story_id_to_metrics:
            all_story_accs.append(all(story_id_to_metrics[story][metric].values()))
        storywise_metrics[metric][fn_noseed_to_idx[fn_noseed]].append(sum(all_story_accs) / len(all_story_accs) * 100)

print(" ==== ")
print(f"{{\"Xs\": {Xs}}}")
for metric in all_metrics:
    print(json.dumps({metric: all_metrics[metric]}))
    # for x_idx, x in Xs:
    print(json.dumps({
        metric: [f"{torch.tensor(all_metrics[metric][x_idx]).mean():.2f} +/- {sem(all_metrics[metric][x_idx]):.2f}" for x_idx, x_metric in enumerate(all_metrics[metric])]
    }))
    print("STORYWISE")
    print(json.dumps({
        metric: [f"{torch.tensor(storywise_metrics[metric][x_idx]).mean():.2f} +/- {sem(storywise_metrics[metric][x_idx]):.2f}" for x_idx, x_metric in enumerate(storywise_metrics[metric])]
    }))
    # storywise_metrics = []
    # for story in story_id_to_metrics:
    #     storywise_metrics.append(all(story_id_to_metrics[story][metric].values()))
    # print(f"storywise {metric}: {torch.tensor(storywise_metrics).float().mean()*100:.2f} +/- {sem(storywise_metrics)*100:.2f}")