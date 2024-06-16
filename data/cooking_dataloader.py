import torch
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader, Dataset, IterableDataset
from utils import DEVICE, pad_stack
import glob
import itertools as it
import random
import string
import pickle as pkl


class CookingState:
    def __init__(self, attribute: str, event_value: list, entity: str=None):
        self.entity = entity
        self.attribute = attribute
        self.event_value = event_value

    def __str__(self):
        if self.entity is None:
            return f"the {self.attribute} is {' and '.join(self.event_value)}"
        else:
            return f"the {self.attribute} of {self.entity} is {' and '.join(self.event_value)}"

class CookingDatasetGPT3(Dataset):
    def __init__(self, data_dir, data_split, start_idx=0, end_idx=-1, train_state=None, data_size=-1):
        self.data_dir = data_dir
        # self.label2stories = {}
        self.data_order_file = os.path.join(data_dir, f"{data_split}_data_order.json")
        if os.path.exists(self.data_order_file):
            self.data_order = json.load(open(self.data_order_file))
        else:
            self.data_order = None
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.train_state = train_state
        self.data_size = data_size
        self.data_split_all_actions_by_recipe = json.load(open(os.path.join(data_dir, f"{data_split}_all_actions.json")))
        # self.data_split_all_actions = set()
        # for recipe in tqdm(self.data_split_all_actions_by_recipe):
        #     self.data_split_all_actions = self.data_split_all_actions.union(set(self.data_split_all_actions_by_recipe[recipe]))

        # build data
        self.data = self.load_data(data_dir, data_split)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]

    def load_data(self, data_dir, data_split):
        num_data = 0
        # recipes2instr = {}
        all_data_processed = {}
        all_actions = {}
        data_split = "test"
        # iterate through recipes
        for fp in tqdm(glob.glob(os.path.join(os.path.join(data_dir, data_split), "*.json"))):
            with open(fp) as f:
                loaded_f = json.load(f)
                all_actions[loaded_f['id']] = set()
                assert loaded_f['split'] == data_split
                # all_instrs_tokenized = []
                # num_context_tokens = 0
                all_ingredients = [ing.replace("_", " ") for ing in loaded_f["ingredient_list"]]
                all_instrs = [f'Ingredients: {", ".join(all_ingredients)}']
                for instr_num in range(len(loaded_f['text'])):
                    instr = ''.join([' ' + word if word not in string.punctuation else word for word in loaded_f['text'][str(instr_num)]]).strip()
                    instr = instr[0].capitalize() + instr[1:]
                    # instr_tokens = self.tokenizer.tokenize(instr)
                    # num_context_tokens += len(instr_tokens)
                    # if num_context_tokens > 2*max_seq_len: break
                    all_instrs.append(instr)
                    # all_instrs_tokenized.append(instr_tokens)
                    all_actions[loaded_f['id']].add(instr)
                # full_data_num_tokens = {'contexts': 0, 'next_instrs': 0, 'states': 0, 'entities': 0}
                # current state of entity -> {attribute -> value}
                curr_entity_to_attr_to_value = {}
                context_sentences = []
                all_valid_actions_list = []
                all_invalid_actions_list = []
                actions_list = []
                states_list = []
                labels = []
                for i in range(1,len(all_instrs)):
                    # if num_data >= max_data_size: break
                    # full_data_num_tokens['contexts'] += len(all_instrs_tokenized[i])
                    # if full_data_num_tokens['contexts'] > max_seq_len: break
                    if str(i) in loaded_f['events']:
                        if 'composition' in loaded_f['events'][str(i)]:
                            del loaded_f['events'][str(i)]['composition']
                        event_dict = loaded_f['events'][str(i)]
                        curr_entities = [loaded_f['ingredient_list'][ing_idx].replace('_', ' ') for ing_idx in loaded_f['ingredients'][str(i)]]
                        # extract propositions
                        propositions = [CookingState(event_type, event_dict[event_type], entity) for event_type, entity in it.product(event_dict, curr_entities)]
                        # integrate new propositions into old ones
                        for prop in propositions:
                            if prop.entity not in curr_entity_to_attr_to_value: curr_entity_to_attr_to_value[prop.entity] = {}
                            curr_entity_to_attr_to_value[prop.entity][prop.attribute] = prop.event_value
                    all_curr_state_full_facts = [[CookingState(
                        attribute=attr, event_value=curr_entity_to_attr_to_value[entity][attr], entity=entity,
                    ) for attr in curr_entity_to_attr_to_value[entity]] for entity in curr_entity_to_attr_to_value]
                    all_curr_state_full_facts = list(it.chain(*all_curr_state_full_facts))
                    states_list.append(all_curr_state_full_facts)
                    context = ' '.join(all_instrs[:i])
                    context_sentences.append(context)
                    all_valid_actions_list.append(all_instrs[i])
                    # randomly sample an action....
                    random_recipes = random.sample(list(self.data_split_all_actions_by_recipe.keys()), 5)
                    possible_negs = []
                    for recipe in random_recipes:
                        for neg_action in self.data_split_all_actions_by_recipe[recipe]:
                            if neg_action not in possible_negs and neg_action != all_instrs[i]:
                                possible_negs.append(neg_action)
                    # breakpoint()
                    # Not *REALLY* negatives but ok...
                    # possible_negs = list(set(self.data_split_all_actions_by_recipe) - {all_instrs[i]})
                    all_invalid_actions_list.append(random.sample(possible_negs, 5))
                breakpoint()
                if len(". ".join(context_sentences)) > 10000:
                    breakpoint()
                data_entry = {
                    'input_sents': context_sentences,
                    # "labels": 'Not OK' if not story['plausible'] and sent_idx >= first_bad_sentence else 'OK',
                    "id": loaded_f['id'],
                    "all_valid_actions": all_valid_actions_list,
                    "all_invalid_actions": all_invalid_actions_list,
                    "states": states_list,
                }
                # if story['example_id'] in all_ids:
                #     # unique-ify
                #     continue
                # entry = {
                #     "input_sents": story['sentences'],
                #     "label_sents": ['Not OK' if not story['plausible'] and sentence_idx >= first_bad_sentence else 'OK' for sentence_idx in range(len(story['sentences']))],
                #     "id": story['example_id'],
                # }
                # all_ids.append(story['example_id'])
                # all_data_processed.append(entry)

                recipe_name = loaded_f['id']
                if recipe_name not in recipes2instr:
                    recipes2instr[recipe_name] = []
                recipes2instr[recipe_name].append(data_entry)
                num_data += 1
        if self.data_order is None:
            random.seed(0)
            random.shuffle(all_ids)
            self.data_order = all_ids
            json.dump(all_ids, open(self.data_order_file, "w"))
        all_data_processed.sort(key=lambda x: self.data_order.index(x['id']))
        if self.end_idx > -1:
            all_data_processed = all_data_processed[:self.end_idx]
        all_data_processed = all_data_processed[self.start_idx:]
        # if self.data_size > -1:
        #     all_data_processed = all_data_processed[:self.data_size]
        # print(f"GPT3 accuracy: {sum(gpt3_accuracy) / len(gpt3_accuracy)}")
        return all_data_processed

class CookingDataset(Dataset):
    _state_effect_to_verb = None
    _states_to_valid_verbs = None
    _all_entities = None
    _all_verbs = None

    def __init__(self, data_dir, tokenizer, data_split, max_seq_len, max_data_size=10000, max_gt_grounded_states=float("inf"), randseed: int = None, control_input: bool = False, **kwargs):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_split = data_split
        self.max_seq_len = max_seq_len
        self.max_data_size = max_data_size
        self.max_gt_grounded_states = max_gt_grounded_states
        self.randseed = randseed
        self.control_input = control_input

        # build data
        self.data = self.load_data(data_dir, data_split, max_seq_len, max_data_size, max_gt_grounded_states)

    @classmethod
    def state_effect_to_verb(cls):
        if CookingDataset._state_effect_to_verb is None:
            # verb -> state change; state change -> verb
            state_change_by_verb = pkl.load(open("cooking_dataset/lexicon/state_change_by_verb_ncl.pickle", "rb"))
            verb_vocab = set([line.strip() for line in open("cooking_dataset/vocabs/verb.vocab").readlines()])
            state_effect_to_verb = {}
            for verb in state_change_by_verb:
                assert verb in verb_vocab
                all_state_effects = []
                for attr in state_change_by_verb[verb]:
                    all_state_effects.append((attr, state_change_by_verb[verb][attr]))
                for state_effect in all_state_effects:
                    if state_effect not in state_effect_to_verb:
                        state_effect_to_verb[state_effect] = []
                    state_effect_to_verb[state_effect].append(verb)
            CookingDataset._state_effect_to_verb = state_effect_to_verb
        return CookingDataset._state_effect_to_verb
    
    @classmethod
    def states_to_valid_verbs(cls):
        if CookingDataset._states_to_valid_verbs is None:
            CookingDataset._states_to_valid_verbs = json.load(open("cooking_dataset/states_to_valid_verbs.json"))
        return CookingDataset._states_to_valid_verbs
    
    @classmethod
    def all_entities(cls):
        if CookingDataset._all_entities is None:
            CookingDataset._all_entities = json.load(open("cooking_dataset/vocabs/ingredients.json"))
        return CookingDataset._all_entities
    
    @classmethod
    def all_verbs(cls):
        if CookingDataset._all_verbs is None:
            CookingDataset._all_verbs = [line.strip() for line in open("cooking_dataset/vocabs/verb.vocab").readlines()]
        return CookingDataset._all_verbs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]

    def load_data(self, data_dir, data_split, max_seq_len, max_data_size, max_num_aligned):
        #{'contexts': [], 'post_contexts': [], 'next_instrs': [], 'states': {'events_facts': [], 'entities_facts': [], 'curr_state_facts': []}, 'filenames': []}
        num_aligned = 0
        num_data = 0
        recipes2instr = {}
        # iterate through recipes
        for fp in tqdm(glob.glob(os.path.join(os.path.join(data_dir, data_split), "*.json"))):
            if num_data >= max_data_size: break
            with open(fp) as f:
                loaded_f = json.load(f)
                assert loaded_f['split'] == data_split
                all_instrs = []
                all_instrs_tokenized = []
                num_context_tokens = 0
                all_ingredients = [ing.replace("_", " ") for ing in loaded_f["ingredient_list"]]
                for instr_num in range(len(loaded_f['text'])):
                    instr = ' '.join(loaded_f['text'][str(instr_num)])
                    if instr_num == 0:
                        instr = f'Ingredients: {", ".join(all_ingredients)} . {instr}'
                    instr_tokens = self.tokenizer.tokenize(instr)
                    num_context_tokens += len(instr_tokens)
                    if num_context_tokens > 2*max_seq_len: break
                    all_instrs.append(instr)
                    all_instrs_tokenized.append(instr_tokens)

                full_data_num_tokens = {'contexts': 0, 'post_contexts': 0, 'next_instrs': 0, 'states': 0, 'entities': 0}
                # current state of entity -> {attribute -> value}
                curr_entity_to_attr_to_value = {}
                for i in range(1,len(all_instrs)):
                    if num_data >= max_data_size: break
                    full_data_num_tokens['contexts'] += len(all_instrs_tokenized[i])
                    full_data_num_tokens['post_contexts'] += len(all_instrs_tokenized[i])
                    if full_data_num_tokens['contexts'] > max_seq_len or full_data_num_tokens['post_contexts'] > max_seq_len: break

                    states_dict = {}
                    # get the entities after the *previous* instruction
                    if str(i-1) in loaded_f['ingredients'] and str(i-1) in loaded_f['events']:
                        curr_entities = [loaded_f['ingredient_list'][ing_idx].replace('_', ' ') for ing_idx in loaded_f['ingredients'][str(i-1)]]
                        states_dict['all_changed_entities_facts'] = curr_entities
                        # states_dict['full_entities_facts'] = 

                        if 'composition' in loaded_f['events'][str(i-1)]:
                            del loaded_f['events'][str(i-1)]['composition']
                        event_dict = loaded_f['events'][str(i-1)]

                        # extract events
                        event_facts = [CookingState(event_type, event_dict[event_type], None) for event_type in event_dict]
                        # TODO better naming...
                        states_dict['all_changed_events_facts'] = event_facts
                        # extract propositions
                        propositions = [CookingState(event_type, event_dict[event_type], entity) for event_type, entity in it.product(event_dict, curr_entities)]
                        states_dict['all_changed_curr_state_facts'] = propositions
                        # integrate new propositions into old ones
                        for prop in propositions:
                            if prop.entity not in curr_entity_to_attr_to_value: curr_entity_to_attr_to_value[prop.entity] = {}
                            curr_entity_to_attr_to_value[prop.entity][prop.attribute] = prop.event_value
                    else:
                        states_dict['all_changed_entities_facts'] = None
                        states_dict['all_changed_events_facts'] = None
                        states_dict['all_changed_curr_state_facts'] = None
                    states_dict['all_curr_state_full_facts'] = [[CookingState(
                        attribute=attr, event_value=curr_entity_to_attr_to_value[entity][attr], entity=entity,
                    ) for attr in curr_entity_to_attr_to_value[entity]] for entity in curr_entity_to_attr_to_value]
                    states_dict['all_curr_state_full_facts'] = list(it.chain(*states_dict['all_curr_state_full_facts']))
                    states_dict['all_entities'] = all_ingredients

                    if num_aligned < max_num_aligned:
                        states_dict['changed_entities_facts'] = states_dict['all_changed_entities_facts']
                        states_dict['changed_events_facts'] = states_dict['all_changed_events_facts']
                        states_dict['changed_curr_state_facts'] = states_dict['all_changed_curr_state_facts']
                        states_dict['curr_state_full_facts'] = states_dict['all_curr_state_full_facts']
                        # states_dict['relevant_state_facts'] = states_dict['all_relevant_state_facts']
                        num_aligned += 1
                    else:
                        states_dict['changed_entities_facts'] = None
                        states_dict['changed_events_facts'] = None
                        states_dict['changed_curr_state_facts'] = None
                        states_dict['curr_state_full_facts'] = None
                        # states_dict['relevant_state_facts'] = None

                    if self.control_input:
                        context = f'Ingredients: {", ".join(all_ingredients)}'
                    else:
                        context = ' '.join(all_instrs[:i])
                    data_entry = {
                        'context': context,
                        'next_instrs': all_instrs[i],
                        'next_verbs': [verb for verb in loaded_f['verb'].get(str(i), []) if verb != "<NO_CHANGE>"],
                        'next_entities': [loaded_f['ingredient_list'][ing_idx].replace('_', ' ') for ing_idx in loaded_f['ingredients'].get(str(i), [])],
                        'post_contexts': all_instrs[i:],
                        'filenames': os.path.split(fp)[-1].replace('.json', ''),
                        'states': states_dict,
                    }
                    recipe_name = loaded_f['id']
                    if recipe_name not in recipes2instr:
                        recipes2instr[recipe_name] = []
                    recipes2instr[recipe_name].append(data_entry)
                    num_data += 1
                    
        # shuffle contexts
        recipes_order = list(recipes2instr.keys())
        if self.randseed:
            random.seed(self.randseed)
            random.shuffle(recipes_order)
        
        # linearize data
        full_data = []
        for recipe in recipes_order:
            for instr_entry in recipes2instr[recipe]:
                full_data.append(instr_entry)
        return full_data


def get_synth_actions(state, gt_next_verb, gt_next_entity, all_valid_entities):  #, entity, state):
    # given a proposition, synthetically generates an invalid action following that proposition
    # clean, comp, cook, shape, temp
    # invalid_actions = False
    # if fact.attribute == "cleanliness":
    #     if fact.event_value == "dry":
    #         invalid_actions = [f"remove water from {fact.entity} ."]
    #     # elif fact.event_value == "dirty":
    #     elif fact.event_value == "clean":
    #         invalid_actions = [f"clean the dirt from {fact.entity} ."]
    # # elif fact.attribute == "composition":
    # elif fact.attribute == "cookedness":
    #     if fact.event_value == "cooked":
    #         invalid_actions = [f"cook {fact.entity} .", f"put {fact.entity} in the oven .", f"put {fact.entity} in the pan ."]
    # elif fact.attribute == "shape":
    #     if fact.event_value == "separated":
    #         invalid_actions = [f"cut {fact.entity} into pieces.", f"separate {fact.entity} ."]
    #     # elif fact.event_value == "deformed":
    #     # elif fact.event_value == "hit":
    #     # elif fact.event_value == "molded":
    # # elif fact.attribute == "temperature":
    # #     if fact.event_value == "hot":
    # #     elif fact.event_value == "cold":
    # #     elif fact.event_value == "room":
    # elif fact.attribute == "location":
    #     invalid_actions = [f"put {fact.entity} in {fact.event_value} .", f"move {fact.entity} to {fact.event_value} ."]
    # CookingDataset.state_effect_to_verb, CookingDataset.states_to_valid_verbs, CookingDataset.all_entities, CookingDataset.all_verbs
    """
    valid action: same verb as GT, but with other fluff removed
    """
    if len(gt_next_verb) > 0 and len(gt_next_entity) > 0:
        try:
            valid_actions = {f"{verb} {entity} ." for verb, entity in it.product(gt_next_verb, gt_next_entity)}
        except:
            import pdb; pdb.set_trace()
    else:
        valid_actions = None
        ent2valid_verbs = {}
    
    """
    invalid action: any action that induces the same change as the current state
    TODO anything else?
    """
    invalid_actions = set()
    if len(state) == 0:
        # generate invalid actions by putting verbs with invalid ingredients
        # TODO should be relatively rare...
        invalid_ingredients = list(set(CookingDataset.all_entities()) - set(all_valid_entities))
        invalid_actions = [f"{verb} {ingredient} ." for verb, ingredient in it.product(random.sample(CookingDataset.all_verbs(), 16), random.sample(invalid_ingredients, 16))]
    for fact in state:
        for event_value in fact.event_value:
            curr_state = (fact.attribute, event_value)
            if curr_state in CookingDataset.state_effect_to_verb():
                invalid_actions = invalid_actions.union({f"{verb} {fact.entity} ." for verb in CookingDataset.state_effect_to_verb()[curr_state]})
            else:
                # invalid actions are actions that don't appear ever with the state in full dataset (TODO only a heuristic, not necessarily valid)
                invalid_verbs = set(CookingDataset.all_verbs()) - set(CookingDataset.states_to_valid_verbs().get(fact.attribute, {}).get(event_value, set()))
                invalid_actions = invalid_actions.union({f"{verb} {fact.entity} ." for verb in invalid_verbs})
        # get a valid state transition from saved
        # valid actions on entity must be consistent with all facts about entity
        if valid_actions is None:
            if fact.entity not in ent2valid_verbs:
                ent2valid_verbs[fact.entity] = None
            for attr_value in fact.event_value:
                if fact.attribute not in CookingDataset.states_to_valid_verbs() or attr_value not in CookingDataset.states_to_valid_verbs()[fact.attribute]: continue
                if ent2valid_verbs[fact.entity] is None:
                    ent2valid_verbs[fact.entity] = set(CookingDataset.states_to_valid_verbs()[fact.attribute][attr_value])
                else:
                    ent2valid_verbs[fact.entity] = ent2valid_verbs[fact.entity].intersection(set(CookingDataset.states_to_valid_verbs()[fact.attribute][attr_value]))

    if valid_actions is None:
        valid_actions = set()
        for ent in ent2valid_verbs:
            valid_actions = valid_actions.union({f"{verb} {ent} ." for verb in ent2valid_verbs[ent]})

    try: assert len(invalid_actions) > 0
    except: import pdb; pdb.set_trace()
    return list(valid_actions), list(invalid_actions)


class CookingDataLoader(DataLoader):
    def __init__(self, dataset: CookingDataset, tokenizer, batch_size, train_state=None, contrastive: str="False"):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        # self.dataset = dataset
        self.train_state = train_state
        self.state_key = None
        if self.train_state:
            self.state_key = self.train_state[-1]
        self.contrastive = 'contrastive' in contrastive
        self.use_synth_negatives = contrastive == 'contrastive_synth'
    
    def collate_fn(self, batch):
        batch_ctxts = self.tokenizer([ex['context'] for ex in batch], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        batch_tgts = self.tokenizer([ex['next_instrs'] for ex in batch], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        batch_filenames = [ex['filenames'] for ex in batch]
        if self.state_key:
            ingredients = [", ".join(ex['states']['all_entities']) for ex in batch]
            batch_tgt_facts = [
                ', '.join([
                    str(state) for state in ex['states'][self.state_key]
                ]) if ex['states'][self.state_key] is not None else '' for i, ex in enumerate(batch)
            ]
            batch_tgt_facts_input = [
                ', '.join([f'Ingredients: {ingredients[i]}'] + [
                    str(state) for state in ex['states'][self.state_key]
                ]) if ex['states'][self.state_key] is not None else '' for i, ex in enumerate(batch)
            ]
            batch_all_tgt_facts = [
                ', '.join(#[f'Ingredients: {ingredients[i]}'] + 
                [
                    str(state) for state in ex['states'][f'all_{self.state_key}']
                ]) if ex['states'][f'all_{self.state_key}'] is not None else '' for i, ex in enumerate(batch)
            ]
            batch_concat_text = [
                f'{ex["context"]} [SEP] {batch_tgt_facts[i]}' if len(batch_tgt_facts) > 0 else ex['context'] for i, ex in enumerate(batch)
            ]
            batch_tgt_state_tok = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_tgt_state_input_tok = self.tokenizer(batch_tgt_facts_input, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_all_tgt_state_tok = self.tokenizer(batch_all_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_concat_text_tok = self.tokenizer(batch_concat_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_tgt_state_ret = {
                self.state_key: batch_tgt_state_tok,
                self.state_key + '_input': batch_tgt_state_input_tok,
                self.state_key + '_gold': batch_all_tgt_state_tok,
                self.state_key + '_concat_text': batch_concat_text_tok,
            }
        else:
            batch_tgt_state_ret = {}

        if self.contrastive:
            if self.use_synth_negatives:
                all_cand_actions = {'input_ids': [], 'attention_mask': []}
                labels = []
                batch_valid_actions = []
                for i, ex in enumerate(batch):
                    # sample fact in state and take something invalid
                    invalid_actions = False
                    valid_actions = False
                    if ex['states']["all_curr_state_full_facts"] is not None:
                        try:
                            random.shuffle(ex['states']["all_curr_state_full_facts"])
                        except: import pdb; pdb.set_trace()
                        valid_actions, invalid_actions = get_synth_actions(ex['states']["all_curr_state_full_facts"], ex['next_verbs'], ex['next_entities'], ex['states']['all_entities'])
                        if len(valid_actions) == 0:
                            # use real action
                            valid_actions = [batch[i]['next_instrs']]
                        valid_actions = random.sample(valid_actions, 1)
                        invalid_actions = random.sample(invalid_actions, min(len(invalid_actions), len(batch)-len(valid_actions)))
                    if not invalid_actions:
                        # haven't found a negative... use random out-of-batch
                        invalid_actions = [batch[(i+1)%len(batch)]['next_instrs']]
                    cand_actions = [
                        #ex['next_instrs'],
                        *valid_actions,
                        *invalid_actions,
                    ]
                    batch_valid_actions.append(random.choice(valid_actions))
                    labels.append([1 for _ in range(len(valid_actions))] + [0 for _ in range(len(invalid_actions))])
                    tokenized_cand_actions = self.tokenizer(cand_actions, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    for k in all_cand_actions:
                        all_cand_actions[k].append(tokenized_cand_actions[k])
                # (bs, n_cands, seqlen)
                batch_tgts['all_cands_input_ids'], batch_tgts['all_cands_attention_mask'] = pad_stack(
                    all_cand_actions['input_ids'], all_cand_actions['attention_mask'], pad_idx=self.tokenizer.pad_token_id, device=DEVICE)
                # (bs, n_cands,)
                # batch_tgts['labels'] = torch.zeros(batch_tgts['all_cands_input_ids'].size()[:2]).to(DEVICE)
                # batch_tgts['labels'][:,0] = 1
                # add padding at end
                batch_tgts['labels'] = torch.tensor([ex_label + [0 for _ in range(batch_tgts['all_cands_input_ids'].size(1)-len(ex_label))] for ex_label in labels]).to(DEVICE)
                # modify training behavior too
                # TODO put this behind a flag
                tokenized_synth_actions = self.tokenizer(batch_valid_actions, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                batch_tgts['input_ids'] = tokenized_synth_actions['input_ids']
                batch_tgts['attention_mask'] = tokenized_synth_actions['attention_mask']
            else:
                # (bs, n_cands, seqlen)
                batch_tgts['all_cands_input_ids'] = batch_tgts['input_ids'].unsqueeze(0).repeat(batch_tgts['input_ids'].size(0),1,1)
                batch_tgts['all_cands_attention_mask'] = batch_tgts['attention_mask'].unsqueeze(0).repeat(batch_tgts['attention_mask'].size(0),1,1)
                # (bs, n_cands,)
                batch_tgts['labels'] = torch.eye(len(batch)).to(DEVICE)
        # (inputs, lang_tgts, init_state, tgt_state, game_ids, entities)
        return batch_ctxts, batch_tgts, None, batch_tgt_state_ret, batch_filenames, None


get_cooking_consistency = None