import os
import argparse

# POST_STATE_MODEL_PATH = "poststate_only_models/pre_bart_lr1e-05_training_traces_tw-treasure_hunter_fact_curr_state_belief_[aligned_data_size]_seed42.p/best_post_state_loss.p"

MAX_DATA_SIZES = {
    'trip': 3238,
    'recipes': 817797,
    'textworld': 32966,
    'openpi': 3194,
}

def get_directory(arch, data_type, exp_type, lang_data_size, aligned_data_size, seed, post_state_setting, em_cycle_period=None, state_key=None, control_input=False):
    # pre_bart_lr1e-05_training_traces_tw-treasure_hunter_@(lang_data_size)_seed@(seed)_sep_action_response
    # pre_bart_lr1e-05_training_traces_tw-treasure_hunter_LM_aux_curr_state_belief_{$lang_data_size}_seed{$seed}_sep_action_response
    # pre_bart_lr1e-05_training_traces_tw-treasure_hunter_LM_aux_curr_state_belief_{$aligned_data_size}_seed{$seed}_ft{$lang_data_size}.{$aligned_data_size}_sep_action_response
    # pre_bart_lr1e-05_training_traces_tw-treasure_hunter_LM_aux_curr_state_belief_{$aligned_data_size}_seed{$seed}_em{$lang_data_size}.{$aligned_data_size}_lm_state_metric_sep_action_response
    assert exp_type in ['lm_only', 'lm_aux_state', 'lang2state', 'state_only', 'lang_state_concat', 'lm_aux_state_ft', 'lm_aux_state_em', 'lm_state_probe', 'lm_state_probe_controlmodel', 'lm_aux_state_em_nopost']
    em_cycle_period_fn = ''
    if em_cycle_period:
        em_cycle_period_fn = f'_cyc{em_cycle_period}'
    aux_fn = ""

    if 'lm_aux_state' in exp_type:
        aux_fn = f"_LM_aux_{state_key}"
    if 'state_only' in exp_type:
        aux_fn = f"_only_fact_{state_key}"
    if 'lang2state' in exp_type or 'lm_state_probe' in exp_type:
        aux_fn = f"_lang_to_{state_key}"
    if 'lang_state_concat' in exp_type:
        aux_fn = f"_concat_fact_{state_key}"
    
    if data_type == "textworld":
        data_path = "training_traces_tw-treasure_hunter"
        suffix = "_sep_action_response"
    elif data_type == "openpi":
        data_path = "openPI_data"
        suffix = ""
    elif data_type == "recipes":
        data_path = "recipes"
        suffix = ""
    elif data_type == "trip":
        data_path = "TRIP_dataset"
        suffix = ""
    else:
        raise NotImplementedError
    
    if aligned_data_size is None:
        aligned_data_size = ''
    elif aligned_data_size < 0:
        aligned_data_str = 'inf'
    else:
        aligned_data_str = f'{aligned_data_size}.{lang_data_size}'

    directory = (
        f"pre_{arch}_lr1e-05_{data_path}"+\
        f"{aux_fn}{'_controlinput' if control_input else ''}"+\
        f"_{lang_data_size}_seed{seed}"+\
        f"{'_ft'+(f'_{post_state_setting}' if post_state_setting else '')+aligned_data_str if exp_type == 'lm_aux_state_ft' else ''}"+\
        f"{'_em'+(f'_{post_state_setting}' if post_state_setting else '')+aligned_data_str if exp_type == 'lm_aux_state_em' else ''}"+\
        f"{'_em_nopost'+(f'_{post_state_setting}' if post_state_setting else '')+f'{aligned_data_size}.{lang_data_size}' if exp_type == 'lm_aux_state_em_nopost' else ''}"+\
        f"{'_probe'+(f'_{post_state_setting}' if post_state_setting else '')+aligned_data_str if exp_type == 'lm_state_probe' else ''}"+\
        f"{'_probe_control'+(f'_{post_state_setting}' if post_state_setting else '') if exp_type == 'lm_state_probe_controlmodel' else ''}"+\
        f"{em_cycle_period_fn}{suffix}"
    )
    return os.path.join('model_checkpoints', directory)


def get_post_state_directory(arch, data_type, aligned_data_size, seed, state_key):
    if data_type == "textworld":
        data_path = "training_traces_tw-treasure_hunter"
        directory = f"pre_{arch}_lr1e-05_{data_path}_concat_fact_{state_key}_{aligned_data_size}_seed{seed}_sep_action_response/lang_models"
        fn = "best_multi_bleu.p"
    else:
        if data_type == "openpi":
            data_path = "openPI_data"
        elif data_type == "recipes":
            data_path = "recipes"
        elif data_type == "trip":
            data_path = "TRIP_dataset"
        else:
            raise NotImplementedError
        directory = f"pre_{arch}_lr1e-05_{data_path}_concat_fact_{state_key}_{aligned_data_size}_seed{seed}/lang_models"
        fn = "best_ex_match_contrastive.p"
    return os.path.join('model_checkpoints', os.path.join(directory, fn))


def copy_checkpoint(source_path, target_path, model_type, extra_warning=""):
    if not os.path.exists(source_path):
        print(f"WARNING: no trained {model_type} model checkpoint found at {source_path}. {extra_warning}")
        successful = False
    else:
        print(f"Loading existing {model_type} model checkpoint from {source_path}")
        cp @(source_path) @(target_path)
        successful = True
    return successful


def main(
    arch, data_type, debug, device, exp_type, metric, post_state_setting, seeds,
    lang_data_sizes, aligned_data_props=None,
    do_train=True, continue_train=False, em_cycle_period=None,
    ensemble_samples=-1, ensemble_weight=0.5,
    eval_type='decoder', state_key=None, control_input=False,
    n_samples=None,
):
    print("======")
    stdout = [f"SWEEP SETTINGS FOR {exp_type}:", f"\tdevice={device}", f"\tseeds={seeds}", f"\tlang_data_sizes={lang_data_sizes}"]
    if aligned_data_props: stdout.append(f"\taligned_data_propotions={aligned_data_props}")
    stdout.append("=======")
    print("\n".join(stdout))
    epochs = 1000
    probe_data_size = MAX_DATA_SIZES[data_type]
    if n_samples is None:
        n_samples = 1 if eval_type == "contrastive" or exp_type in ["lang2state", "lm_state_probe", "lm_state_probe_controlmodel"] else 5
    if eval_type == "decoder":
        fn = f"best_{metric}.p"
    else:
        fn = f"best_{metric}_{eval_type}.p"
    for seed in seeds:
        for lang_data_size in lang_data_sizes:
            if not aligned_data_props:
                aligned_data_sizes = [None]  # dummy value
            else:
                aligned_data_sizes = [int(aligned_data_prop*lang_data_size) for aligned_data_prop in aligned_data_props]

            for aligned_data_size in aligned_data_sizes:
                # compute patience
                if 'probe' in exp_type: data_size = probe_data_size
                else: data_size = lang_data_size
                eval_interval = -1
                if data_size > 10000:
                    if data_type == 'recipes':
                        patience = 5
                        eval_interval = 2500
                    elif data_type == 'textworld':
                        patience = 5
                        eval_interval = 1000
                elif data_size >= 5000: patience = 5
                elif data_size >= 1000: patience = 10
                elif data_size >= 100: patience = 15
                else: patience = 20

                if exp_type in ['lm_state_probe', 'lm_state_probe_controlmodel']:
                    aligned_data_size = probe_data_size

                # make directory
                directory = get_directory(arch, data_type, exp_type, lang_data_size, aligned_data_size, seed, post_state_setting, em_cycle_period, state_key=state_key, control_input=control_input)

                print("\n\n======")
                print(f"NEW EXPERIMENT: {lang_data_size} lang-only examples, {aligned_data_size} aligned examples, seed {seed} - patience {str(patience*eval_interval)+' updates' if eval_interval > -1 else str(patience)+' epochs'}")
                print(f"{directory}")
                if do_train and not continue_train:
                    num = 1
                    while os.path.exists(os.path.join(directory, f"lang_models/{fn[:-2]}{n_samples}_samples.jsonl")):
                        directory_new = f"{directory}{num}" 
                        print(f"WARNING: directory already contains trained checkpoint, renaming directory {directory} -> {directory_new}")
                        directory = directory_new
                        num += 1
                if continue_train:
                    assert os.path.exists(directory)
                has_prev_directory = os.path.exists(directory)
                mkdir -p @(directory)/lang_models
                print("======")

                train_state = f'LM_aux_{state_key}'
                state_train_state = f'fact_{state_key}'
                
                copied_checkpoint = True
                # specify experiment-specific arguments/make experiment-specific directories
                if exp_type == 'state_only':
                    train_state = train_state.replace('LM_aux_', 'only_fact_')
                elif exp_type == 'lang2state' or 'lm_state_probe' in exp_type:
                    train_state = train_state.replace('LM_aux_', 'lang_to_')
                    metric = 'fact_f1'
                elif exp_type == 'lang_state_concat':
                    train_state = train_state.replace('LM_aux_', 'concat_fact_')
                if exp_type == 'lm_only':
                    exp_args = f"--train_data_size {lang_data_size} --force_overwrite_checkpoint"
                elif exp_type in ['state_only', 'lang2state', 'lang_state_concat', 'lm_aux_state']:
                    if do_train:
                        mkdir -p @(directory)/state_models
                    exp_args = f"--train_data_size {lang_data_size} --train_state {train_state}"
                elif exp_type == 'lm_state_probe_controlmodel':
                    exp_args = f"--train_data_size {probe_data_size} --train_state {train_state} --no_pretrain"
                elif exp_type == 'lm_state_probe':
                    if do_train and not continue_train:
                        mkdir -p @(directory)/state_models
                        base_fn = "best_multi_bleu.p" if data_type == 'textworld' else "best_ex_match_contrastive.p"
                        src = os.path.join(get_directory(
                            arch, data_type, "lm_only", lang_data_size, aligned_data_size, seed, post_state_setting, None, state_key=state_key, control_input=control_input,
                        ), f"lang_models/{base_fn}")
                        tgt = os.path.join(directory, f"lang_models/{fn}")
                        copied_checkpoint = copy_checkpoint(src, tgt, "lang")
                    exp_args = f"--train_data_size {probe_data_size} --train_state {train_state}"
                elif exp_type == 'lm_aux_state_ft': #or exp_type == 'lm_state_probe':
                    if do_train and not continue_train:
                        mkdir -p @(directory)/state_models
                        src = os.path.join(get_directory(
                            arch, data_type, "lm_only", lang_data_size, aligned_data_size, seed, post_state_setting, None, state_key=state_key, control_input=control_input,
                        ), f"lang_models/{fn}")
                        tgt = os.path.join(directory, f"lang_models/{fn}")
                        copied_checkpoint = copy_checkpoint(src, tgt, "lang")
                    # if exp_type == 'lm_aux_state_ft':
                    exp_args = f"--train_data_size {lang_data_size} --train_state {train_state} --max_gt_grounded_states {aligned_data_size} --force_overwrite_checkpoint"
                    # elif exp_type == 'lm_state_probe':
                    #     exp_args = f"--train_data_size {probe_data_size} --train_state {train_state}"
                elif exp_type == 'lm_aux_state_em' or exp_type == 'lm_aux_state_em_nopost':
                    if do_train and not continue_train:
                        mkdir -p @(directory)/state_models
                        mkdir -p @(directory)/post_state_models
                        lm_aux_state_ft_directory = get_directory(arch, data_type, "lm_aux_state_ft", lang_data_size, aligned_data_size, seed, post_state_setting, None, state_key=state_key, control_input=control_input)
                        copied_checkpoint = copy_checkpoint(os.path.join(lm_aux_state_ft_directory, f"lang_models/{fn}"), os.path.join(directory, f"lang_models/{fn}"), "lang", "Falling back to lang-only checkpoint")
                        copied_checkpoint &= copy_checkpoint(os.path.join(lm_aux_state_ft_directory, f"state_models/{fn}"), os.path.join(directory, f"state_models/{fn}"), "state", "Falling back to lang-only checkpoint")
                        if not copied_checkpoint:
                            lang_only_directory = get_directory(arch, data_type, "lm_only", lang_data_size, aligned_data_size, seed, post_state_setting, None, state_key=state_key, control_input=control_input)
                            copy_checkpoint(os.path.join(lang_only_directory, f"lang_models/{fn}"), os.path.join(directory, f"lang_models/{fn}"), "lang")
                        if aligned_data_size > 0 and (post_state_setting == 'post_state_model' or exp_type == 'lm_aux_state_em'):
                            post_state_model_path = get_post_state_directory(arch, data_type, aligned_data_size, seed, state_key=state_key)
                            # POST_STATE_MODEL_PATH.replace('[aligned_data_size]', str(aligned_data_size))
                            copied_checkpoint &= copy_checkpoint(post_state_model_path, os.path.join(directory, f"post_state_models/{fn}"), "post-state")
                    if not em_cycle_period:
                        em_cycle_period_arg = int(patience / 2)
                    else:
                        em_cycle_period_arg = em_cycle_period
                    print(f"EM period: {em_cycle_period_arg}")
                    exp_args = f"--train_data_size {lang_data_size} --train_state {train_state} --max_gt_grounded_states {aligned_data_size} --force_overwrite_checkpoint --do_state_em --eval_state_em --em_cycle_period {em_cycle_period_arg}"
                    if exp_type == 'lm_aux_state_em': exp_args += f" --post_state_setting post_state_model"
                else:
                    raise NotImplementedError()
                if 'lm_state_probe' in exp_type:
                    exp_args += f' --probe_encoder'
                if debug:
                    exp_args += f" --debug"
                    print("DEBUG MODE")
                if ensemble_samples != -1:
                    assert not do_train
                    # 160/num_samples/ensemble_samples
                    if eval_type == "contrastive":
                        eval_batchsize = 10 // ensemble_samples
                    else:
                        eval_batchsize = 32 // ensemble_samples
                    exp_args += f" --ensemble_samples {ensemble_samples} --ensemble_weight {ensemble_weight}"
                else:
                    if eval_type == "contrastive":
                        eval_batchsize = 10
                    else:
                        if data_type == "openpi": eval_batchsize = 8
                        else: eval_batchsize = 16
                state_exp_args = []
                if post_state_setting:
                    exp_args += f" --post_state_setting {post_state_setting}"
                    state_exp_args += [f"--post_state_setting", f"{post_state_setting}"]
                print("======\n")
                if not debug and not copied_checkpoint:
                    print(f"Couldn't find checkpoint {copied_checkpoint}, skipping.")
                    continue

                lm_dir = os.path.join(directory, 'lang_models')
                log_path = os.path.join(directory, 'train.log')
                exp_args = exp_args.split(' ')

                if data_type == "openpi":
                    data_specific_args = ["--data", "openPI_data", "--data_type", "openpi"]
                    assert eval_type == "contrastive" or exp_type in ["lang2state", "lm_state_probe", "lm_state_probe_controlmodel"]
                elif data_type == "recipes":
                    data_specific_args = ["--data", "cooking_dataset/recipes", "--data_type", "recipes"]
                    assert eval_type == "contrastive" or exp_type in ["lang2state", "lm_state_probe", "lm_state_probe_controlmodel"]
                elif data_type == "trip":
                    data_specific_args = ["--data", "TRIP_dataset", "--data_type", "trip"]
                    assert eval_type == "contrastive" or exp_type in ["lang2state", "lm_state_probe", "lm_state_probe_controlmodel"]
                elif data_type == "textworld":
                    data_specific_args = [
                        "--data", "tw_games/training_traces_tw-treasure_hunter",
                        "--gamefile", "tw_games/training_tw-treasure_hunter",
                        "--data_type", "textworld",
                    ]
                else:
                    raise NotImplementedError()
                lm_data_specific_args = data_specific_args
                if arch.strip() == 'bart':
                    arch_name = 'bart-base'
                elif arch.strip() == 't5':
                    arch_name = 't5-base'
                else:
                    arch_name = arch
                    exp_args += ['--max_seq_len', 1024 if 'bart' in arch else 512]
                
                if control_input:
                    exp_args += ['--control_input']

                # train command
                if do_train:
                    env CUDA_VISIBLE_DEVICES=@(device) python train_lm.py \
                        @(lm_data_specific_args) \
                        --arch @(arch_name) \
                        --epochs @(epochs) \
                        --seed @(seed) \
                        --patience @(patience) \
                        --metric @(metric) \
                        --save_dir @(lm_dir) \
                        --local_files_only \
                        --eval_batchsize @(eval_batchsize) \
                        --eval_type @(eval_type) \
                        --eval_interval @(eval_interval) \
                        @(exp_args) \
                        2>&1 | tee -a @(log_path)
                if os.path.exists(log_path):
                    # eval command
                    env CUDA_VISIBLE_DEVICES=@(device) python train_lm.py \
                        @(lm_data_specific_args) \
                        --arch @(arch_name) \
                        --epochs @(epochs) \
                        --seed @(seed) \
                        --patience @(patience) \
                        --metric @(metric) \
                        --save_dir @(lm_dir) \
                        --local_files_only \
                        --num_samples @(n_samples) \
                        --eval_batchsize @(eval_batchsize) \
                        --eval_type @(eval_type) \
                        @(exp_args) \
                        --eval_only


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_type', type=str, choices=[
        'lm_only', 'lm_aux_state', 'state_only', 'lang2state', 'lang_state_concat', 'lm_aux_state_ft', 'lm_aux_state_em', 'lm_aux_state_em_nopost', 'lm_state_probe', 'lm_state_probe_controlmodel',
    ])
    parser.add_argument('--arch', type=str, default='bart', choices=['bart', 'bart-base', 'bart-large', 't5', 't5-base', 't5-large'])
    parser.add_argument('--control_input', action='store_true', default=False, help="only condition on initial state with entities")
    parser.add_argument('--device', '-d', type=int, default=0)
    parser.add_argument('--data_type', type=str, choices=['textworld', 'openpi', 'recipes', 'trip'])
    parser.add_argument('--state_key', type=str, choices=[
        'curr_state_belief', 'curr_state', 'relevant_state', 'full', 'curr_state_full', 'belief', 'preconditions', 'effects', 'preconditions_effects'
    ], default=None, help='use state_key other than default for epxeriment_type and data_type')
    parser.add_argument('--seeds', '-s', type=str, default="3,4,5,6")
    parser.add_argument('--lang_data_sizes', type=str, default="100,1000,4000,10000")
    parser.add_argument('--aligned_data_props', type=str, default="0.01,0.05,0.1,0.5")
    parser.add_argument('--metric', type=str, default='multi_bleu')
    parser.add_argument('--post_state_setting', type=str, choices=[None, 'post_state_model', 'early_fusion', 'late_fusion'], default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_type', type=str, choices=['contrastive', 'decoder'], default='decoder')
    parser.add_argument('--continue_train', action='store_true', default=False)
    parser.add_argument('--em_cycle_period', type=int, default=None)
    parser.add_argument('--ensemble_samples', type=int, default=-1)
    parser.add_argument('--ensemble_weight', type=float, default=0.5)
    parser.add_argument('--n_samples', type=int, default=None)

    args = parser.parse_args()
    exp_type = args.experiment_type
    seeds = args.seeds.split(',')
    lang_data_sizes = args.lang_data_sizes.split(',')
    aligned_data_props = args.aligned_data_props.split(',')
    metric = args.metric
    seeds = [int(s) for s in seeds]
    lang_data_sizes = [int(ds) for ds in lang_data_sizes]
    state_key = args.state_key
    if state_key is None:
        if args.data_type == 'textworld':
            state_key = 'curr_state_belief'
        elif args.data_type == 'openpi':
            state_key = 'curr_state'  # ?
        elif args.data_type == 'recipes':
            state_key = 'curr_state_full'  # ?
        elif args.data_type == 'trip':
            state_key = 'relevant_state'  # ?
    if exp_type == 'lm_only' or exp_type == 'lm_aux_state' or exp_type == 'state_only' or exp_type == 'lang_state_concat' or exp_type == 'lang2state' or exp_type == 'lm_state_probe_controlmodel':
        aligned_data_props = None
    elif exp_type == 'lm_aux_state_ft' or exp_type == 'lm_aux_state_em' or exp_type == 'lm_aux_state_em_nopost' or exp_type == 'lm_state_probe':
        aligned_data_props = [float(ds) for ds in aligned_data_props]
        if exp_type == 'lm_state_probe':
            aligned_data_props = [1.0]
    else:
        raise NotImplementedError()

    main(
        arch=args.arch, data_type=args.data_type, debug=args.debug, device=args.device,
        exp_type=exp_type, metric=metric, post_state_setting=args.post_state_setting,
        seeds=seeds, lang_data_sizes=lang_data_sizes, aligned_data_props=aligned_data_props,
        do_train=(not args.eval_only),
        continue_train=args.continue_train, em_cycle_period=args.em_cycle_period, ensemble_samples=args.ensemble_samples,
        ensemble_weight=args.ensemble_weight, eval_type=args.eval_type, state_key=state_key, control_input=args.control_input,
        n_samples=args.n_samples,
    )
