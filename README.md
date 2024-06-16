# Latent States

## Setup
```
conda create -n sitsup PYTHON=3.7
pip install -r requirements.txt
```

You may need to run
```bash
export PYTHONPATH=.
```
before the below commands.


## TW Fine-tuning
```bash
xonsh scripts/lm_training_sweep.py [lm_only|lm_aux_state_ft|lm_aux_state_em] --arch [bart-base|bart-large] --data_type textworld -d [cuda_device] -s [seed1,seed2,...] --lang_data_sizes 1000 [--aligned_data_props 0.5] [--eval_lang_only]
```
* `lm_only`: language-only fine-tuning
* `lm_aux_state_ft`: auxiliary supervision
    * `aligned_data_props` must be set in this setting
* `lm_aux_state_em`: latent state supervision
    * `aligned_data_props` must be set in this setting
* set `--eval_lang_only` to evaluate the final trained LM (first run without this flag, then run with this flag.)


## TRIP Fine-tuning
### Language-only Training
```bash
# 1. train
python clean_train.py --arch bart-base --train_data_size 1000 --seed $s
# 2. evaluate
python clean_train.py --arch bart-base --train_data_size 1000 --seed $s --eval_only
```

### Auxiliary Supervision
**Full State Supervision (all training samples are annotated with state)**
```bash
python clean_train.py --arch bart-base --train_data_size 1000 --train_state LM_aux_gpt3_state --seed $s [--eval_only]
```

**Partial State Supervision (only some training samples are annotated with state)**
```bash
# Using gpt3-generated state
python clean_train.py --arch bart-base --train_state LM_aux_gpt3_state --metric lm_em --train_data_size 1000 --state_data_size 500 --seed $s [--eval_only]
# Using original state
python clean_train.py --arch bart-base --train_state LM_aux_relevant_state --metric lm_em --train_data_size 1000 --state_data_size 500 --seed $s [--eval_only]
```
Saves under `model_checkpoints_new/bart-base_lr1e-05_TRIP_LM_aux_relevant_state_ft_100.1000_seed{$s}`

### Latent State
1. Train p(next_sentence | state) model
```bash
python clean_train.py --arch bart-base --train_state relevant_state_to_lang --metric lm_em --train_data_size 500 --seed $s [--eval_only]
```
Saves under `model_checkpoints_new/bart-base_lr1e-05_TRIP_relevant_state_to_lang_ft_500.1000_seed{$s}`

2. Train auxiliary supervision model to convergence on partial dataset
See guidance under **Auxiliary Supervision** / **Partial State Supervision**.

3. Train latent state
```bash
# train
python clean_train.py --arch bart-base --train_state LM_aux_relevant_state --metric lm_em --train_data_size 1000 --state_data_size 500 --seed $s --state_em_cycle_period 5 --lmauxstate_load_path model_checkpoints_new/bart-base_lr1e-05_TRIP_LM_aux_relevant_state_ft_500.1000_seed{$s} --poststate_load_path model_checkpoints_new/bart-base_lr1e-05_TRIP_relevant_state_to_lang_ft_500_seed{$s}/lang_models/best_lm_em.p
# evaluate
python clean_train.py --arch bart-base --train_state LM_aux_relevant_state --metric lm_em --train_data_size 1000 --state_data_size 500 --seed $s --state_em_cycle_period 5 --eval_only
```


## TW/TRIP Prompting
```bash
# set openai key
env OPENAI_API_KEY=<OPENAI_API_KEY>
# query lm
python query_lm.py --data_type [textworld|TRIP] (--with_state [our|orig]) (--latent_state 5) --eval_type [generate|classify] --num_samples [1|5]
```
* `eval_type` should be `generate` for textworld, `classify` for TRIP
* Include `--with_state` tag for auxiliary supervision or latent supervision, and specify type of the auxiliary state: ours (handcrafted + generated with GPT) or orig (original states associated dataset)
* Include `--latent_state` tag for latent supervision, then specify the number of candidate latent states to generate / rerank.
* To reproduce paper results: `num_samples` should be 5 for textworld, 1 for TRIP


## Print Results
Get results reported in the paper from saved evaluation files.
Current paper reports *Precision* of 5 samples for TW results, and *Story-wise Correctness* for TRIP results.
- Finetuning
```bash
# In TW
python eval_scripts/get_metrics.py --domain textworld --exp [lang_only|lang_state_ft|lang_state_em]
python eval_scripts/get_metrics_TRIP.py --exp [lang_only|LM_aux_state_full|LM_aux_state_partial|lang_state_em]
```

- Prompting
```bash
python eval_scripts/query_comparable.py
```

