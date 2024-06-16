import openai
import json
from transformers import T5TokenizerFast, GPT2TokenizerFast
from data.trip_dataloader import TRIPDatasetGPT3, TRIPDataset
from data.dataloader import load_data
from data.tw_dataloader import TWDatasetGPT3
from data.cooking_dataloader import CookingDatasetGPT3
import os
from tqdm import tqdm
import random
import numpy as np
import argparse
import torch.nn.functional as F
import torch
import glob
import textworld
from retry import retry
from utils import closestSum


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, choices=["TRIP", "textworld", "recipes"], default='TRIP')
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--comparable_state_tokens', action='store_true', default=False, help="get # of lang exs as comparable to prompt with state")
parser.add_argument('--prompt_with_state', type=str, choices=["our", "orig", False], default=False, help="only prompt with state (not inference)")
parser.add_argument('--with_state', type=str, choices=["our", "orig", False], default=False)
parser.add_argument('--latent_state', type=int, default=None, help="How many candidates to generate for latent state. 0 for no latent state.")
parser.add_argument('--with_explanation', action='store_true', default=False)
parser.add_argument('--bad_sentence', action='store_true', default=False)
parser.add_argument('--data_start_idx', type=int, default=0)
parser.add_argument('--data_end_idx', type=int, default=-1)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--state_to_next_sentence', action='store_true', default=False)  #?
parser.add_argument('--eval_type', type=str, default='classify', choices=['classify', 'generate', 'gpt3'])
args = parser.parse_args()


openai.api_key = os.getenv("OPENAI_API_KEY")
data_type = args.data_type
gpt3_model = "text-davinci-002"
# gpt3_model = "code-davinci-002"
gpt3_file = f"gpt3-{gpt3_model}/{data_type}/{gpt3_model if gpt3_model != 'text-davinci-002' else 'gpt3'}_cache.jsonl"
gpt3_file = f"gpt3-{gpt3_model}/{data_type}/{gpt3_model if gpt3_model != 'text-davinci-002' else 'gpt3'}_cache_temp_0.jsonl"
starter_prompt_file = None
if args.latent_state is not None:
    starter_prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_with_{args.with_state}state_LS.txt'
    output_fn = f"gpt3-{gpt3_model}/{data_type}/with_latent_{args.with_state}state_{args.latent_state}cands_outs{args.num_samples}.jsonl"
    prompt_file = output_fn.replace(f"_outs{args.num_samples}.jsonl", "_prompt.jsonl")
elif args.bad_sentence:
    # find the single bad sentence
    prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_badsentence.txt'
    output_fn = f"gpt3-{gpt3_model}/{data_type}/badsentence_outs.jsonl"
elif args.with_explanation and args.with_state:
    prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_with_{args.with_state}state_explain.txt'
    output_fn = f"gpt3-{gpt3_model}/{data_type}/{args.eval_type}_with_{args.with_state}state_explain_outs{args.num_samples}.jsonl"
elif args.with_explanation:
    prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_explain.txt'
    output_fn = f"gpt3-{gpt3_model}/{data_type}/{args.eval_type}_explain_outs{args.num_samples}.jsonl"
else:
    if args.prompt_with_state:
        prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_with_{args.prompt_with_state}state.txt'
    elif args.comparable_state_tokens:
        prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_textonly_comparable.txt'
    else:
        prompt_file = f'gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_textonly.txt'
    output_fn = f"gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_with_{args.prompt_with_state}states{'_comparable' if args.comparable_state_tokens else ''}_out_with_{args.with_state}states{args.num_samples}.jsonl"

os.makedirs(os.path.split(gpt3_file)[0], exist_ok=True)
cache = {}
if os.path.exists(gpt3_file):
    with open(gpt3_file) as f:
        all_cached_lines = f.readlines()
        for item in tqdm(all_cached_lines, desc="Loading GPT3 cache"):
            item = json.loads(item)
            cache[item['prompt']] = item['result']


def save_gpt3_result(gpt3_file, new_results):
    with open(gpt3_file, "a") as wf:
        for prompt in new_results:
            wf.write(json.dumps({"prompt": prompt, "result": new_results[prompt]}) + "\n")


@retry(openai.error.RateLimitError, delay=3, backoff=1.5, max_delay=20, tries=10)
def openai_completion_query(**kwargs):
    return openai.Completion.create(**kwargs)


def gpt3_score_prompt(engine, input_prefix, classes, cache=None):
    new_cache_results = {}
    class_scores = []
    # optimal_class = None
    for cl in classes:
        input_str = input_prefix + cl
        if cache and input_str in cache:
            result = cache[input_str]
        else:
            result = openai_completion_query(
                engine=engine,
                prompt=input_str,
                max_tokens=0,
                logprobs=0,
                echo=True,
            )
            new_cache_results[input_str] = result
            cache[input_str] = result
        save_gpt3_result(gpt3_file, new_cache_results)
        for token_position in range(len(result['choices'][0]['logprobs']['tokens'])):
            if ''.join(
                result['choices'][0]['logprobs']['tokens'][:token_position]
            ).strip() == input_prefix.strip():
                break
        score = sum(result['choices'][0]['logprobs']['token_logprobs'][token_position:]) / len(result['choices'][0]['logprobs']['token_logprobs'][token_position:])
        class_scores.append(score)
    return class_scores, new_cache_results


def gpt3_generate(engine, input_prefix, cache=None, temperature=0.7, n_samples=1):
    new_cache_results = {}
    class_scores = []
    input_str = input_prefix
    if temperature == 0 and cache and input_str in cache:
        result = cache[input_str]
    else:
        result = openai_completion_query(
            engine=engine,
            prompt=input_str,
            max_tokens=256,
            logprobs=0,
            echo=True,
            stop=["\n", "<|endoftext|>"],
            temperature=temperature,
            n=n_samples,
        )
        new_cache_results[input_str] = result
        cache[input_str] = result
    if temperature == 0:
        save_gpt3_result(gpt3_file, new_cache_results)
    choice_to_score = {}
    choice_to_score2 = {}
    for choice in result['choices']:
        index_of_prefix_token = choice['logprobs']['text_offset'].index(len(input_prefix))
        gen_scores = choice['logprobs']['token_logprobs'][index_of_prefix_token:]
        gen_tokens = choice['logprobs']['tokens'][index_of_prefix_token:]
        if "<|endoftext|>" in gen_tokens:
            eot_token_idx = gen_tokens.index("<|endoftext|>")
            gen_scores = gen_scores[:eot_token_idx]
            gen_tokens = gen_tokens[:eot_token_idx]
        if "\n" in gen_tokens:
            nl_token_idx = gen_tokens.index("\n")
            gen_scores = gen_scores[:nl_token_idx]
            gen_tokens = gen_tokens[:nl_token_idx]
        gen_score = sum(gen_scores) / len(gen_scores)
        generation = ''.join(gen_tokens)
        choice_to_score[generation] = gen_score
        choice_to_score2[generation] = sum(gen_scores)
    return result['choices'][0]['text'][len(input_prefix):], choice_to_score, new_cache_results


def get_best_facts(curr_prompt, next_generation_to_score, ncands_to_gen, cache=None, fact_prefix="\nKnown facts:", next_generation_suffix="\n"):
    """
    get best facts consistent with current prompt so far and next generation
    """
    fact_sample_temperature = 0 if ncands_to_gen == 1 else 0.9
    fact_candidates, facts_to_precedingprob, new_results = gpt3_generate(
        gpt3_model, "\n".join(curr_prompt) + fact_prefix,
        cache=cache, temperature=fact_sample_temperature, n_samples=ncands_to_gen,
    )
    nextgen_prob_to_preceding_facts = {}
    fact_candidates_to_preceding_next_probs = {}
    for fact_candidate in facts_to_precedingprob:
        # score next sentence
        next_line_score, new_results = gpt3_score_prompt(
            gpt3_model, "\n".join(curr_prompt) + fact_prefix + fact_candidate + "\n" + next_generation_suffix, [next_generation_to_score], cache=cache,
        )
        nextgen_prob_to_preceding_facts[next_line_score[0] + facts_to_precedingprob[fact_candidate]] = fact_candidate
        fact_candidates_to_preceding_next_probs[fact_candidate] = [facts_to_precedingprob[fact_candidate], next_line_score[0]]
    bestgen_prob = max(list(nextgen_prob_to_preceding_facts))
    best_facts = nextgen_prob_to_preceding_facts[bestgen_prob]
    return best_facts, fact_candidates_to_preceding_next_probs


tokenizer = T5TokenizerFast.from_pretrained('t5-base')
def get_prompt(args, prompt_file, starter_prompt_file):
    make_prompt = False
    if os.path.exists(prompt_file):
        prompt_exs = open(prompt_file).read()
        if args.latent_state is not None and args.latent_state > 0 and os.path.exists(starter_prompt_file):
            # remove candidates
            new_exs = []
            for line in prompt_exs.split("\n"):
                if not line[:len("Candidates: ")] == "Candidates: ":
                    new_exs.append(line)
            prompt_exs = "\n".join(new_exs)
    elif args.latent_state is not None and args.latent_state == 0 and os.path.exists(starter_prompt_file):
        prompt_exs = open(starter_prompt_file).read()
    elif args.latent_state is not None and args.latent_state > 0 and os.path.exists(starter_prompt_file):
        fact_prefix = "\nKnown facts:"
        prompt_exs = open(starter_prompt_file).read()
        all_lines = prompt_exs.split("\n")
        curr_prompt = []
        if not args.debug:
            wf = open(prompt_file, "w")
        for ln in range(len(all_lines)):
            if not args.debug:
                wf.flush()
                wf.write(all_lines[ln]+"\n")
            curr_prompt.append(all_lines[ln])
            if ln == len(all_lines) - 2:
                break
            if args.data_type == "TRIP" and (not all_lines[ln+1].startswith("Known facts: ") and all_lines[ln+1].strip() != "" and all_lines[ln] == "OK"):
                best_facts, fact_candidates_to_preceding_next_probs = get_best_facts(curr_prompt, all_lines[ln+2], args.latent_state, cache, fact_prefix, next_generation_suffix=all_lines[ln+1]+"\n")
                if not args.debug:
                    wf.write(fact_prefix.lstrip() + best_facts+"\n")
                curr_prompt.append(fact_prefix.lstrip() + best_facts)
                wf.write(f"Candidates: "+ json.dumps(fact_candidates_to_preceding_next_probs) + "\n")
            elif args.data_type == "textworld" and (not all_lines[ln].startswith("Known facts: ") and all_lines[ln+1].startswith(">") and all_lines[ln+1] != "> inventory"):
                best_facts, fact_candidates_to_preceding_next_probs = get_best_facts(curr_prompt, all_lines[ln+1] + "\n" + all_lines[ln+2], args.latent_state, cache, fact_prefix)
                if not args.debug:
                    wf.write(fact_prefix.lstrip() + best_facts+"\n")
                curr_prompt.append(fact_prefix.lstrip() + best_facts)
                wf.write(f"Candidates: "+ json.dumps(fact_candidates_to_preceding_next_probs) + "\n")
            else:
                continue
        if not args.debug:
            wf.close()
        prompt_exs = "\n".join(curr_prompt)
        # with open(prompt_file, "w") as wf:
        #     wf.write(prompt_exs)
    elif args.data_type == "textworld":
        gpttokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        for fn in glob.glob(os.path.join("tw_games/training_tw-treasure_hunter", 'train/*.ulx')):
            env = textworld.start(fn)
            game_state = env.reset()
            game_kb = game_state['game'].kb.inform7_predicates
            inform7_game = env._inform7
            break
        train_dataset = TWDatasetGPT3("tw_games/training_traces_tw-treasure_hunter", "train", start_idx=0, end_idx=-1, tokenizer=gpttokenizer, inform7_game=inform7_game, debug=args.debug)
        examples = open(f"gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_textonly.txt").read().strip()
        if args.comparable_state_tokens:
            curr_n_tokens = len(gpttokenizer.tokenize(examples))
            tgt_n_tokens = len(gpttokenizer.tokenize(
                open(f"gpt3-{gpt3_model}/{data_type}/{args.eval_type}_prompt_with_origstate.txt").read().strip()
            ))
            prompt_additional_tokens = tgt_n_tokens - curr_n_tokens
        examples_sofar = examples.split("\n\n")

        prompt_exs = []
        n_prompt_tokens_so_far = []
        for data_idx, entry in enumerate(train_dataset):
            inputs_so_far = []
            story_so_far = []
            tokens_so_far = 0
            for sentence_idx, sentence in enumerate(entry["input_sents"]):
                tokens_so_far += len(gpttokenizer.tokenize(sentence))
                inputs_so_far.append(sentence)
                story_so_far.append(sentence)
                if args.with_state == "orig":
                    tokens_so_far += len(gpttokenizer.tokenize("Known facts: " + ". ".join(entry["states"][sentence_idx])+".\n"))
                    inputs_so_far.append("Known facts: " + ". ".join(entry["states"][sentence_idx])+".\n")
            prompt_ex = "".join(inputs_so_far).strip()
            n_prompt_tokens_so_far.append(len(gpttokenizer.tokenize(prompt_ex)))
            prompt_exs.append(prompt_ex)
        if args.comparable_state_tokens:
            additional_prompt_exs = closestSum(n_prompt_tokens_so_far, prompt_additional_tokens)
        make_prompt = True
    elif args.data_type == "TRIP":
        gpttokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        if args.with_state == "orig":
            tokenizer = T5TokenizerFast.from_pretrained('t5-base')
            train_dataset = TRIPDataset('TRIP_dataset', tokenizer, 'train', 2048, contrastive=False, train_state=["only_fact", "relevant_state_facts"], control_input=False)
            context_to_states = {}
            for data in train_dataset:
                if data['states'].get('all_relevant_state_facts', None) is not None:
                    context_to_states[data['context'].strip()] = ". ".join(data['states']['all_relevant_state_facts']) + "."
            context_to_ids = {}
            for data in train_dataset:
                context_to_ids[data['context'].strip()] = data['idx']
        else:
            train_dataset = TRIPDatasetGPT3('TRIP_dataset', 'train', )
            # story2idx = {' '.join(data['context']): data['idx'] for data in train_dataset}
        # items_per_label = 8
        examples = open(f"gpt3-{gpt3_model}/{data_type}/classify_prompt_textonly.txt").read().strip()
        if args.comparable_state_tokens:
            curr_n_tokens = len(gpttokenizer.tokenize(examples))
            tgt_n_tokens = len(gpttokenizer.tokenize(
                open(f"gpt3-{gpt3_model}/{data_type}/classify_prompt_with_ourstate.txt").read().strip()
            ))
            prompt_additional_tokens = tgt_n_tokens - curr_n_tokens
        examples = examples.split("\n\n")[1:]
        # examples = []
        prompt_exs = []
        n_prompt_tokens_so_far = []
        for data in train_dataset:
            inputs_so_far = []
            for sentence_idx, sentence in enumerate(data['input_sents']):
                inputs_so_far.append(sentence)
                inputs_so_far.append(data['label_sents'][sentence_idx])
            prompt_tokens = "\n".join(inputs_so_far) + "\n\n"
            prompt_exs.append(prompt_tokens)
            if prompt_tokens in examples: continue
            n_prompt_tokens_so_far.append(len(gpttokenizer.tokenize(prompt_tokens)+1))
            # if sum(n_prompt_tokens_so_far) >= TRIP_prompt_tokens:
            #     break
        if args.comparable_state_tokens:
            additional_prompt_exs_ntoks = closestSum(n_prompt_tokens_so_far, prompt_additional_tokens-1)
            additional_prompt_exs = []
            for ntok in additional_prompt_exs_ntoks:
                additional_prompt_exs.append(prompt_exs[n_prompt_tokens_so_far.index(ntok)])
        print("".join(additional_prompt_exs).strip())
        make_prompt = True
    elif args.data_type == "recipes":
        train_dataset = CookingDatasetGPT3("cooking_dataset/recipes", "train")
        make_prompt = True
    if make_prompt:
        prompt_exs = "\n\n".join(prompt_exs)
        with open(prompt_file, "w") as wf:
            wf.write(prompt_exs)
    print(prompt_exs)
    return prompt_exs


prompt_exs = get_prompt(args, prompt_file, starter_prompt_file)
print(f"Output file: {output_fn}")
output_file = open(output_fn, "w")


classes = ["Not OK", "OK"]
positive_class = classes[1]
per_sentence_metrics = {
    "accuracy": 0,
    "cm": torch.tensor([[0,0],[0,0]]).long(),
    "gt_label_prob": torch.tensor([[0,0],[0,0]]).float(),
}
per_story_metrics = {
    "accuracy": 0,
    "cm": torch.tensor([[0,0],[0,0]]).long(),
    "gt_label_prob": torch.tensor([[0,0],[0,0]]).float(),
}
# 1598
data_start_idx = args.data_start_idx
data_end_idx = args.data_end_idx
if args.data_type == "textworld":
    for fn in glob.glob(os.path.join("tw_games/training_tw-treasure_hunter", f'{args.split}/*.ulx')):
        env = textworld.start(fn)
        game_state = env.reset()
        game_kb = game_state['game'].kb.inform7_predicates
        inform7_game = env._inform7
        break
    dev_dataset = TWDatasetGPT3("tw_games/training_traces_tw-treasure_hunter", args.split, start_idx=0, end_idx=100 if args.debug else -1, tokenizer=tokenizer, inform7_game=inform7_game)
elif args.data_type == "TRIP":
    dev_dataset = TRIPDatasetGPT3('TRIP_dataset', args.split, data_start_idx, data_end_idx)
elif args.data_type == "recipes":
    dev_dataset = CookingDatasetGPT3("cooking_dataset/recipes", args.split, data_start_idx, data_end_idx)
pbar = tqdm(dev_dataset)
n_sentences = 0
n_exs = 0
example_id_to_sentence_to_gen_facts = {}
data_idx = 0
for data_point in pbar:
    if args.debug and n_exs > 50:
        break
    pred_label = None
    prompt_so_far = []
    if args.with_state:
        assert data_point['id'] not in example_id_to_sentence_to_gen_facts
        example_id_to_sentence_to_gen_facts[data_point['id']] = {}
    output_file.write(f"{n_exs}. {data_point['id']}\n")
    for sentence_idx, sentence in enumerate(data_point['input_sents']):
        if args.data_type == "TRIP":
            prompt_so_far.append(sentence)
            output_file.write(sentence+"\n")
            true_label = int(classes[1].strip() == data_point['label_sents'][sentence_idx].strip())
            if sentence_idx == 0:
                prompt_so_far.append("OK")
                pred_label = 1
                correct = true_label == pred_label
                output_file.write("OK " + ("X" if not correct else "") + "\n")
            elif pred_label is not None and classes[pred_label] == "Not OK":
                prompt_so_far.append("Not OK")
                pred_label = 0
                correct = true_label == pred_label
                output_file.write("Not OK " + ("X" if not correct else "") + "\n")
                continue
            else:
                binary_scores, new_results = gpt3_score_prompt(
                    gpt3_model, prompt_exs + "\n\n" + "\n".join(prompt_so_far) + "\n", classes, cache=cache,
                )
                pred_label = int(binary_scores[0] < binary_scores[1])
                prompt_so_far.append(classes[pred_label])
                correct = true_label == pred_label
                output_file.write(classes[pred_label] + " " + ("X" if not correct else "") + "\n")

                probabilities = F.softmax(torch.tensor(binary_scores))

                per_sentence_metrics['cm'][true_label, pred_label] += 1
                per_sentence_metrics['gt_label_prob'][true_label] += probabilities
                per_sentence_metrics['accuracy'] += correct
                n_sentences += 1
        else:
            if sentence_idx > 7: break
            if sentence_idx > 0:
                # # generate following from previous sentence...
                try:
                    generation, choice_to_score, new_results = gpt3_generate(
                        gpt3_model, prompt_exs + "\n\n" + "\n".join(prompt_so_far) + "\n>", cache=cache,
                        temperature=(0 if args.num_samples == 1 else 0.7), n_samples=args.num_samples,
                    )
                except openai.error.InvalidRequestError:
                    # too long
                    break
                correct = 0.0
                for generation in choice_to_score:
                    action = "> " + generation.strip()
                    last_correct = action in data_point["all_valid_actions"][sentence_idx-1]
                    correct += last_correct
                    output_file.write(f"Generation ({'X' if not last_correct else ''}): "+action +"\n")
                correct /= len(choice_to_score)
                output_file.write(f"All valid actions: {json.dumps(data_point['all_valid_actions'][sentence_idx-1])}\n")
                output_file.write(f"All invalid actions: {json.dumps(data_point['all_invalid_actions'][sentence_idx-1])}\n")
                per_sentence_metrics['accuracy'] += correct
                n_sentences += 1
            prompt_so_far.append(sentence.strip())
            output_file.write(sentence+"")

        if args.with_state or args.with_explanation or args.bad_sentence:
            if args.data_type == "TRIP" and (args.bad_sentence or args.with_explanation) and classes[pred_label] == "Not OK":
                if args.bad_sentence:
                    prefix = "\nContradicting sentence:"
                elif args.with_explanation:
                    prefix = "\nContradicting facts:"
            elif args.with_state and (args.data_type != "TRIP" or (classes[pred_label] == "OK" and sentence_idx != len(data_point['input_sents']) - 1)):
                prefix = "\nKnown facts:"
            else:
                output_file.flush()
                continue
            try:
                generation, _, new_results = gpt3_generate(
                    gpt3_model, prompt_exs + "\n\n" + "\n".join(prompt_so_far) + prefix, cache=cache, temperature=0,
                )
            except openai.error.InvalidRequestError:
                # too long
                break
            if args.with_state:
                if prefix == "\nKnown facts:":
                    generation_key = "pred_facts"
                else:
                    generation_key = "bad_facts"
                example_id_to_sentence_to_gen_facts[data_point['id']][sentence_idx] = {generation_key: generation}
                if args.data_type == "TRIP":
                    example_id_to_sentence_to_gen_facts[data_point['id']][sentence_idx]["pred_label"] = classes[pred_label]
                    example_id_to_sentence_to_gen_facts[data_point['id']][sentence_idx]["true_label"] = classes[true_label]
            prompt_so_far.append(prefix.strip() + generation)  #'. '.join(generated_facts))
            output_file.write(prefix.strip() + generation+"\n")
        output_file.flush()
    n_exs += 1
    output_file.write("\n\n")

    if args.data_type == "TRIP":
        true_label = int(classes[1].strip() == data_point['label_sents'][-1].strip())
        correct = true_label == pred_label
        per_story_metrics['cm'][true_label, pred_label] += 1
        per_story_metrics['accuracy'] += correct
        pbar.set_description(f"story_acc={per_story_metrics['accuracy'] / (data_point['idx'] + 1)}")
    else:
        pbar.set_description(f"sentence_acc={per_sentence_metrics['accuracy'] / n_sentences}")

output_file.close()
if len(example_id_to_sentence_to_gen_facts) > 0:
    json.dump(example_id_to_sentence_to_gen_facts, open(f"gpt3-{gpt3_model}/TRIP_{args.split}_gen_facts_{data_start_idx}-{data_end_idx}.json", "w"), indent=4)
print("PER-SENTENCE METRICS")
print(f"acc = {per_sentence_metrics['accuracy'] / n_sentences}")
print(f"cm = {per_sentence_metrics['cm']}")
print(f"avg_correct_prob = {per_sentence_metrics['gt_label_prob'] / per_sentence_metrics['gt_label_prob'].sum(-1).unsqueeze(-1)}")

print("PER-STORY METRICS")
print(f"acc = {per_story_metrics['accuracy'] / n_exs}")
print(f"cm = {per_story_metrics['cm']}")
print(f"avg_correct_prob = {per_story_metrics['gt_label_prob'] / per_story_metrics['gt_label_prob'].sum(-1).unsqueeze(-1)}")