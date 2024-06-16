import torch
from tqdm import tqdm
import json
import os
import copy
from torch.utils.data import DataLoader, Dataset, IterableDataset
from models.models import ContrastiveClassifierHead, JointClassifierHead
from data.TRIP_data_utils.prepro import get_tiered_data
from data.TRIP_data_utils.ann import att_to_num_classes, idx_to_att, att_default_values
from collections import Counter
from utils import DEVICE, pad_stack
import glob
import itertools as it
import random
import inflect
import regex as re
import copy
inflect = inflect.engine()

entity_value_dict = {
    'h_location': {
        1: 'disappeared',
        2: 'moved',
    },
    'location': {
        1: 'is disappeared',
        2: 'is picked up',
        3: 'is put down',
        4: 'is put on',
        5: 'is removed',
        6: 'is put in container',
        7: 'is taken out of container',
        8: 'is moved',
    },
}

att_to_idx = {'h_location': 0, 
              'conscious': 1, 
              'wearing': 2, 
              'h_wet': 3, 
              'hygiene': 4, 
              'location': 5, 
              'exist': 6, 
              'clean': 7, 
              'power': 8, 
              'functional': 9, 
              'pieces': 10, 
              'wet': 11, 
              'open': 12, 
              'temperature': 13, 
              'solid': 14, 
              'contain': 15, 
              'running': 16, 
              'moveable': 17, 
              'mixed': 18, 
              'edible': 19}
idx_to_att = {v: k for k,v in att_to_idx.items()}
human_atts = ['h_location', 'conscious', 'wearing', 'h_wet', 'hygiene']
att_to_num_classes = {
    "h_location": 3,
    "conscious": 9,
    "wearing": 9,
    "h_wet": 9,
    "hygiene": 9,
    "location": 9,
    "exist": 9,
    "clean": 9,
    "power": 9,
    "functional": 9,
    "pieces": 9,
    "wet": 9,
    "open": 9,
    "temperature": 9,
    "solid": 9,
    "contain": 9,
    "running": 9,
    "moveable": 9,
    "mixed": 9,
    "edible": 9
}

# location_att_to_nl = {
#     "h_location": [None, "[E] disappeared", "The location of [E] moved"],
#     "location": [
#         None, "The [E] disappeared", "The [E] [is] picked up", "The [E] [is] put down", "The [E] [is] put on", "The [E] [is] removed",
#         "The [E] [is] in a container", "The [E] [is] outside a container", "The location of the [E] moved",
#     ],
# }
# TODO agreement, tense?
# nonlocation_att_to_false_true_nl = {
#     "conscious": ["unconscious", "conscious"],
#     "wearing": ["unclothed", "clothed"],
#     "h_wet": ["dry", "wet"],
#     "hygiene": ["dirty", "clean"],
#     "exist": ["not exist", "exist"],  # verb is different
#     "power": ["powered off", "powered on"],
#     "functional": ["not functional", "functional"],
#     "pieces": ["whole", "in pieces"],
#     "wet": ["dry", "wet"],
#     "open": ["closed", "open"],
#     "temperature": ["cold", "hot"],
#     "solid": ["one solid", "not one solid"],  #?
#     "contain": ["empty", "not empty"],
#     "running": ["not running", "running"],
#     "moveable": ["not moveable", "moveable"],
#     "mixed": ["not mixed", "mixed"],
#     "edible": ["inedible", "edible"]
# }
att_change_dir = {'h_location': {0: 'does not move to a new location', 1: 'disappears', 2: 'moves somewhere new'},
            'location': {0: 'does not move to a new location', 1: 'disappears', 2: 'is picked up', 3: 'is put down', 4: 'is put on', 5: 'is removed', 6: 'is put into a container', 7: 'is taken out of a container', 8: 'moved somewhere new'},
            'default': {0: (-1,-1), 1: (0, 0), 2: (1, 1), 3: (1, 0), 4: (0, 1), 5: (-1, 0), 6: (-1, 1), 7: (0, -1), 8: (1, -1)}}
att_adj = { 'conscious': ('unconscious', 'conscious'),
            'wearing': ('undressed', 'dressed'), 
            'h_wet': ('dry', 'wet'), 
            'hygiene': ('dirty', 'clean'), 
            'exist': ('nonexistent', 'existent'), 
            'clean': ('dirty', 'clean'),
            'power': ('unpowered', 'powered'), 
            'functional': ('broken', 'functional'), 
            'pieces': ('whole', 'in pieces'), 
            'wet': ('dry', 'wet'), 
            'open': ('closed', 'open'), 
            'temperature': ('cold', 'hot'), 
            'solid': ('fluid', 'solid'), 
            'contain': ('empty', 'occupied'), 
            'running': ('turned off', 'turned on'), 
            'moveable': ('stuck', 'moveable'), 
            'mixed': ('separated', 'mixed'), 
            'edible': ('inedible', 'edible')}
verb_to_plurality_tense = {
    "is": {
        "singular": {"past": "was", "present": "is"},
        "plural": {"past": "were", "present": "are"},
    },
    "does": {
        "singular": {"past": "did", "present": "does"},
        "plural": {"past": "did", "present": "do"},
    },
}
attr_value_to_nl_template = [
    None,
    [False, False],
    [True, True],
    [True, False],
    [False, True],
    [None, False],
    [None, True],
    [False, None],
    [True, None],
    # "[E] [V/past] previously [att/False] and [V/present] now [att/False]",
    # "[E] [V/past] previously [att/True] and [V/present] now [att/True]",
    # "[E] [V/past] previously [att/True] and [V/present] now [att/False]",
    # "[E] [V/past] previously [att/False] and [V/present] now [att/True]",
    # "[E] [V/present] now [att/False]",
    # "[E] [V/present] now [att/True]",
    # "[E] [V/past] previously [att/False]",
    # "[E] [V/past] previously [att/True]",
]


def attribute_value_to_nl(entity_name: str, attr_name: str, attr_value: int, is_human: bool):
    """
    nlify attribute and value
    """
    is_plural = inflect.singular_noun(entity_name)
    if attr_name in att_change_dir:
        nl_string_predicates = att_change_dir[attr_name][attr_value]
    elif attr_name in att_adj:
        nl_string_segments = []
        attr_value = attr_value_to_nl_template[attr_value]
        # if attr_name == "exist":
        #     verb = verb_to_plurality_tense['does']['plural' if is_plural else 'singular']
        #     if attr_value[0] is not None:
        #         nl_attr_value_past = att_adj[attr_name][attr_value[0]]
        #         nl_string_segments += [f"previously {verb['past']} {nl_attr_value_past}"]
        #     if attr_value[1] is not None:
        #         nl_attr_value_present = att_adj[attr_name][attr_value[1]]
        #         nl_string_segments += [f"now {verb['present']} {nl_attr_value_present}"]
        # else:
        if attr_value[0] != attr_value[1] and attr_value[0] is not None and attr_value[1] is not None:
            # 3,4
            verb = "became"
            nl_attr_value_present = att_adj[attr_name][attr_value[1]]
            nl_string_predicates = f"{verb} {nl_attr_value_present}"
        elif attr_value[1] is None:
            # 7,8
            verb = verb_to_plurality_tense['is']['plural' if is_plural else 'singular']['past']
            nl_attr_value_past = att_adj[attr_name][attr_value[0]]
            nl_string_predicates = f"{verb} {nl_attr_value_past}"
        else:
            # 1,2,5,6
            verb = verb_to_plurality_tense['is']['plural' if is_plural else 'singular']['present']
            nl_attr_value_present = att_adj[attr_name][attr_value[1]]
            nl_string_predicates = f"{verb} {nl_attr_value_present}"
        # if attr_value[0] is not None:
        #     nl_attr_value_past = att_adj[attr_name][attr_value[0]]
        #     nl_string_segments += [f"{verb['past']} previously {nl_attr_value_past}"]
        # if attr_value[1] is not None:
        #     nl_attr_value_present = att_adj[attr_name][attr_value[1]]
        #     nl_string_segments += [f"{verb['present']} now {nl_attr_value_present}"]
        # nl_string_predicates = " and ".join(nl_string_segments)
    else:
        assert False
    if attr_name in human_atts or entity_name[0].isupper() or is_human:
        nl_string = f"{entity_name} {nl_string_predicates}"
    else:
        nl_string = f"The {entity_name} {nl_string_predicates}"
    return nl_string


def convert_attr_vector_to_nl(entity_name, is_actor, entity_state_vector):
    nl_facts = []
    for attr_num, attr_value in enumerate(entity_state_vector):
        attr_value = int(attr_value)
        # skip irrelevant ones
        if attr_value == 0: continue
        attr_name = idx_to_att[attr_num]
        nl_facts.append(attribute_value_to_nl(entity_name, attr_name, attr_value, is_actor))
    return nl_facts

def convert_attr_vector_to_dict(entity_state_vector):
    fact_dict = {}
    for attr_num, attr_value in enumerate(entity_state_vector):
        attr_value = int(attr_value)
        # skip irrelevant ones
        attr_name = idx_to_att[attr_num]
        fact_dict[attr_name] = attr_value
    return fact_dict


def convert_entity_state_to_nl(entity_state, sent_idx, humans):
    """
    entity_state
    sent_idx: idx of last sentence in context
    """
    entity_name = entity_state['entity']
    is_human = entity_name in humans
    nlfied_entity_state = []
    # ['example_id', 'base_id', 'sentences', 'entity', 'attributes', 'preconditions', 'effects', 'conflict_span', 'conflict_span_onehot', 'plausible', 'span_labels']
    for attr_num, attr_value in enumerate(entity_state['attributes'][sent_idx]):
        attr_value = int(attr_value)
        # skip irrelevant ones
        if attr_value == 0: continue
        attr_name = idx_to_att[attr_num]
        nlfied_entity_state.append(attribute_value_to_nl(entity_name, attr_name, attr_value, is_human))
    return nlfied_entity_state
    

class TRIPDatasetGPT3(Dataset):
    def __init__(self, data_dir, data_split, start_idx=0, end_idx=-1, sentence_wise=False, train_state=None, data_size=-1, state_data_size=-1, seed=0):
        self.data_dir = data_dir
        # self.label2stories = {}
        self.seed = seed
        self.data_order_file = os.path.join(data_dir, f"{data_split}_data_order{'_sentence' if sentence_wise else ''}_seed{seed}.json")
        if os.path.exists(self.data_order_file):
            self.data_order = json.load(open(self.data_order_file))
        else:
            self.data_order = None
        
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.sentence_wise = sentence_wise
        self.facts_file = f"gpt3-text-davinci-002/TRIP_old/TRIP_{data_split}_gen_facts.json"
        self.facts = {}
        if os.path.exists(self.facts_file):
            self.facts = json.load(open(self.facts_file))
        self.train_state = train_state
        self.data_size = data_size
        self.state_data_size = state_data_size

        # build data
        self.data = self.load_data(data_dir, data_split, sentence_wise)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]

    def load_data(self, data_dir, data_split, sentence_wise: bool = False):
        num_aligned = 0

        with open(os.path.join(data_dir, 'www_2s_new.json'), 'r') as f:
            # what does cloze / order mean???
            cloze_dataset_2s, order_dataset_2s = json.load(f)
        
        all_data = cloze_dataset_2s[data_split]
        tiered_data = get_tiered_data(cloze_dataset_2s)
        # random.seed(0)
        # random.shuffle(all_data)

        all_data_processed = []
        full_data_num_tokens = {'contexts': 0, 'post_contexts': 0, 'next_instrs': 0, 'states': 0, 'entities': 0}
        all_ids = []
        verified_entries = set()
        gpt3_correct_entries = set()
        entries_with_facts = set()
        gpt3_accuracy = []
        # iterate through story pairs
        for example in tqdm(all_data):
            data_entry = {}
            assert len(example['stories']) == 2   # pairs of valid/invalid stories
            
            # establishes setting / preliminary info
            entities = example['stories'][0]['objects'].split(', ')
            pos_idx = example['label']
            assert len(example['stories'][pos_idx]['states']) == example['length']

            for story in example['stories']:
                if not story['plausible']:
                    first_bad_sentence = min([pair[-1] for pair in story['confl_pairs']])

                if self.sentence_wise:
                    context_so_far = []
                    fact_history = []
                    for sent_idx, sentence in enumerate(story['sentences']):
                        facts = None
                        sample_id = story['example_id'] + "_" + str(sent_idx)
                        if sample_id in all_ids:
                            # unique-ify
                            continue
                        if self.train_state:
                            if "relevant_state" in self.train_state:
                                all_relevant_state_facts = []
                                for attr_name in story['states'][sent_idx]:
                                    for entity_value in story['states'][sent_idx][attr_name]:
                                        attr_value = entity_value[1]
                                        entity_name = entity_value[0]
                                        if attr_value == 0: continue
                                        is_actor = entity_name == story['actor']
                                        ent_fact = attribute_value_to_nl(entity_name, attr_name, attr_value, is_actor)
                                        all_relevant_state_facts.append(ent_fact)
                                facts = ". ".join(all_relevant_state_facts)
                                verified_entries.add(sample_id)
                            elif "gpt3_state" in self.train_state:
                                if not story['plausible'] and sent_idx >= first_bad_sentence:
                                    facts = "Not OK"
                                elif len(self.facts[story['example_id']][str(sent_idx)]["pred_facts_supervision"]) > 0:
                                    facts = self.facts[story['example_id']][str(sent_idx)]["pred_facts_supervision"].strip()
                                if self.facts[story['example_id']]["verified"]:
                                    verified_entries.add(sample_id)
                                elif self.facts[story['example_id']]['gpt3_correct']:
                                    gpt3_correct_entries.add(sample_id)
                            fact_history.append(facts)
                        context_so_far.append(sentence)
                        if None in fact_history:
                            facts = None
                        entry = {
                            "input_sents": copy.deepcopy(context_so_far),
                            "labels": 'Not OK' if not story['plausible'] and sent_idx >= first_bad_sentence else 'OK',
                            "id": sample_id,
                            "facts": facts,
                            "facts_gold": facts,
                            "fact_history": copy.deepcopy(fact_history),
                            "prev_facts": fact_history[-2] if len(fact_history) >= 2 else None,
                            "prev_facts_gold": fact_history[-2] if len(fact_history) >= 2 else None,
                        }
                        gpt3_accuracy.append((entry["facts"] == "Not OK" and entry["labels"] == "Not OK") or (entry["facts"] != "Not OK" and entry["labels"] != "Not OK"))
                        if facts is not None or entry['labels'] == 'Not OK':
                            entries_with_facts.add(sample_id)
                        all_ids.append(sample_id)
                        all_data_processed.append(entry)
                else:
                    if story['example_id'] in all_ids:
                        # unique-ify
                        continue
                    entry = {
                        "input_sents": story['sentences'],
                        "label_sents": ['Not OK' if not story['plausible'] and sentence_idx >= first_bad_sentence else 'OK' for sentence_idx in range(len(story['sentences']))],
                        "id": story['example_id'],
                    }
                    all_ids.append(story['example_id'])
                    all_data_processed.append(entry)
                # if story['plausible'] not in self.label2stories:
                #     self.label2stories[story['plausible']] = []
                # self.label2stories[story['plausible']].append(entry)
        if self.data_order is None:
            random.seed(self.seed)
            random.shuffle(all_ids)
            self.data_order = all_ids
            json.dump(all_ids, open(self.data_order_file, "w"))
        all_data_processed.sort(key=lambda x: self.data_order.index(x['id']))
        if self.end_idx > -1:
            all_data_processed = all_data_processed[:self.end_idx]
        all_data_processed = all_data_processed[self.start_idx:]
        if self.data_size > -1:
            # """
            # filter by verified/gpt3 correct first
            all_sample_ids_to_use = verified_entries
            curr_n_without_facts = 0
            all_data_processed_filtered = [entry for entry in all_data_processed if entry['id'] in verified_entries]
            if len(all_data_processed_filtered) < self.data_size:
                all_data_processed_filtered.extend([entry for entry in all_data_processed if entry['id'] in gpt3_correct_entries and entry['id'] not in verified_entries])
            if len(all_data_processed_filtered) < self.data_size:
                all_data_processed_filtered.extend([entry for entry in all_data_processed if entry['id'] in entries_with_facts and entry['id'] not in gpt3_correct_entries and entry['id'] not in verified_entries])
            all_data_processed = all_data_processed_filtered[:self.data_size]
        if self.state_data_size > -1:
            for i, data_entry in enumerate(all_data_processed):
                if i >= self.state_data_size:
                    data_entry["facts"] = None
                    data_entry["prev_facts"] = None
            # """
            # all_data_processed = all_data_processed[:self.data_size]
        # print(f"GPT3 accuracy: {sum(gpt3_accuracy) / len(gpt3_accuracy)}")
        return all_data_processed
    
    def set_expected_states(self, expected_states):
        assert len(expected_states) == len(self)
        for eidx, entry in enumerate(expected_states):
            self.data[eidx]['facts_expected'] = expected_states[eidx]  #.split('. ')


class TRIPDataLoaderGPT3(DataLoader):
    def __init__(self, dataset: TRIPDatasetGPT3, tokenizer, batch_size, device, decoder_inputs=None, input_type: str="lang"):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.device = device
        self.decoder_inputs = decoder_inputs
        self.input_type = input_type
    
    def collate_fn(self, input_batch):
        # entry['fact_history']
        if self.input_type == "lang":
            contexts = self.tokenizer([' [SEP] '.join([
                ' '.join(entry['input_sents'][:-1]) if len(entry['input_sents']) >= 2 else "",
                entry['input_sents'][-1],
            ]) for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device)
        elif self.input_type == "state":
            contexts = self.tokenizer([' [SEP] '.join([
                entry['prev_facts'] if entry['prev_facts'] is not None else "",
                entry['input_sents'][-1],
            ]) for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device)
        else:
            assert False
        labels = self.tokenizer([entry['labels'] for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device)
        facts_input = []
        # facts_gold = []
        # facts_gold_mask = []
        # prev_facts_gold = []
        facts_mask = []
        fact_history = []
        for entry in input_batch:
            has_prev_facts = entry['prev_facts'] is not None
            facts_mask.append(has_prev_facts)
            # facts_gold.append(entry["facts_gold"])
            # prev_facts_gold.append(entry['fact_history'][-2] if len(entry['fact_history']) >= 2 else "")
            if has_prev_facts:
                facts_input.append(entry['prev_facts'] + " [SEP] " + entry['input_sents'][-1])
                fact_history.append([val for pair in zip(entry['input_sents'], entry['fact_history']) for val in pair])
                # entry['input_sents'][sent_idx] + entry['fact_history'][sent_idx])
        # try:
        #     self.tokenizer([entry['facts_gold'] for entry in input_batch], return_tensors='pt', padding=True, truncation=True)
        # except:
        #     breakpoint()
        batch = {
            # main
            "contexts": contexts,
            "labels": labels,
            "ids": [entry['id'] for entry in input_batch],
            # for latent
            "prev_contexts": self.tokenizer([' '.join(entry['input_sents'][:-1]) for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device),
            "curr_sentences": self.tokenizer([entry['input_sents'][-1] for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device),
            "prev_facts_gold": self.tokenizer([entry['prev_facts_gold'] if entry['prev_facts_gold'] is not None else "" for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device),
            # extra
            # "facts_gold": self.tokenizer([entry['facts_gold'] for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device),
            "input_sents": [entry["input_sents"] for entry in input_batch],
            "fact_history": [entry["fact_history"] for entry in input_batch],
        }
        if "facts_expected" in input_batch[0]:
            batch["facts_expected_input"] = self.tokenizer([' [SEP] '.join([
                entry['facts_expected'],
                entry['input_sents'][-1],
            ]) for entry in input_batch], return_tensors='pt', padding=True, truncation=True).to(self.device)
        if len(facts_input) > 0:
            batch["facts_input"] = self.tokenizer(facts_input, return_tensors='pt', padding=True, truncation=True).to(self.device)
            batch["facts_mask"] = torch.tensor(facts_mask).bool().to(self.device)
            if self.decoder_inputs is not None:
                if self.decoder_inputs == "state_history":
                    # inputs exclude the last decision
                    try:
                        batch["decoder_inputs"] = self.tokenizer([' | '.join(fh[:-1]) + " | " for fh in fact_history], return_tensors='pt', padding=True, truncation=True).to(self.device)
                        batch["decoder_outputs"] = self.tokenizer([' | '.join(fh) for fh in fact_history], return_tensors='pt', padding=True, truncation=True).to(self.device)
                    except:
                        breakpoint()
        return batch


class TRIPDataset(Dataset):
    def __init__(self, data_dir, tokenizer, data_split, max_seq_len, max_data_size=10000, max_gt_grounded_states=float("inf"), randseed: int = None, contrastive: bool = False, train_state = None, control_input: bool = False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_split = data_split
        self.max_seq_len = max_seq_len
        self.max_data_size = max_data_size
        self.max_gt_grounded_states = max_gt_grounded_states
        self.randseed = randseed
        self.contrastive = contrastive
        self.classify = True
        self.train_state = train_state
        self.state_key = None
        if self.train_state:
            self.state_key = self.train_state[-1]
        self.control_input = control_input
        self.facts = json.load(open(f"gpt3-text-davinci-002/TRIP_old/TRIP_{data_split}_gen_facts.json"))

        # build data
        self.data = self.load_data(data_dir, data_split, max_seq_len, max_data_size, max_gt_grounded_states)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]

    def load_data(self, data_dir, data_split, max_seq_len, max_data_size, max_num_aligned):
        # example: 'example_id' (str), 'stories', 'length' (int), 'label' (int 0/1), 'breakpoint' (int 0/1), 'confl_sents' (List[int]), 'confl_pairs' (List[Tuple[int]])
        # stories: ['story_id', 'worker_id', 'type', 'idx', 'aug', 'actor', 'location', 'objects', 'sentences', 'length', 'example_id', 'plausible', 'breakpoint', 'confl_sents', 'confl_pairs', 'states']
        num_aligned = 0

        with open(os.path.join(data_dir, 'www_2s_new.json'), 'r') as f:
            # what does cloze / order mean???
            cloze_dataset_2s, order_dataset_2s = json.load(f)
        
        all_data = cloze_dataset_2s[data_split]
        tiered_data = get_tiered_data(cloze_dataset_2s)
        if self.randseed:
            random.seed(self.randseed)
            random.shuffle(all_data)
        
        # for p in cloze_dataset_2s:
        #     label_dist = Counter([ex['label'] for ex in cloze_dataset_2s[p]])
        #     print('Cloze label distribution (%s):' % p)
        #     print(label_dist.most_common())
        # print_dict(cloze_dataset_2s['train'][0])
        # tiered_dataset = get_tiered_data(cloze_dataset_2s)

        all_data_processed = []
        full_data_num_tokens = {'contexts': 0, 'post_contexts': 0, 'next_instrs': 0, 'states': 0, 'entities': 0}
        # iterate through story pairs
        for example in tqdm(all_data):
            if len(all_data_processed) >= max_data_size: break
            assert len(example['stories']) == 2   # pairs of valid/invalid stories
            # TODO get all states and continuations (optionally: invalid continuation)
            
            # establishes setting / preliminary info
            entities = example['stories'][0]['objects'].split(', ')
            if self.control_input:
                prelim_sent = f"{example['stories'][0]['actor']} is in {example['stories'][0]['location']}. {example['stories'][0]['actor']} can see: {example['stories'][0]['objects']}."
            else:
                prelim_sent = ""
            pos_idx = example['label']
            assert len(example['stories'][pos_idx]['states']) == example['length']
            # try:
            #     assert example['confl_pairs'] == [sorted([confl_sent, example['breakpoint']]) for confl_sent in example['confl_sents']]
            # except:
            #     import pdb; pdb.set_trace()
            # TODO ALL dev examples
            # if self.contrastive and (min(example['confl_sents']) > example['breakpoint'] or example['stories'][0]['sentences'][:example['breakpoint']] != example['stories'][1]['sentences'][:example['breakpoint']]):
            #     # negative is wrt prior state...
            #     continue

            # sentence that negative is ok up to: first sentence that is conflicted by an earlier sentence
            neg_valid_breakpoint = min([max(confl_pair) for confl_pair in example['confl_pairs']])

            for sent_idx in range(example['length']-1):
                if self.contrastive and sent_idx != example['breakpoint'] - 1:  #example['stories'][pos_idx]['sentences'][sent_idx+1] != example['stories'][1-pos_idx]['sentences'][sent_idx+1]:
                    # only query at breakpoint (when pos/neg should have meaningfully different results)
                    continue
                # `sent_idx` tracks last sentence in the context
                if self.control_input:
                    prev_context_pos = []
                    prev_context_neg = []
                else:
                    prev_context_pos = example['stories'][pos_idx]['sentences'][:sent_idx+1]
                    prev_context_neg = example['stories'][1-pos_idx]['sentences'][:sent_idx+1]
                prev_context_pos = [prelim_sent] + prev_context_pos
                prev_context_neg = [prelim_sent] + prev_context_neg
                pos_next_sent = example['stories'][pos_idx]['sentences'][sent_idx+1]
                neg_next_sent = example['stories'][1-pos_idx]['sentences'][sent_idx+1]
                post_context_pos = example['stories'][pos_idx]['sentences'][sent_idx+1:]
                post_context_neg = example['stories'][1-pos_idx]['sentences'][sent_idx+1:]

                if self.contrastive and self.state_key == 'preconditions_facts' and pos_next_sent == neg_next_sent:
                    # is a difference in the preceding context
                    # won't show up in state as will be using exclusively positive states' preconditions
                    continue
                # else:
                #     if self.contrastive: continue
                #     neg_next_sent = None
                #     prev_context_neg = prev_context_pos
                """
                Load state
                """
                story_idxs_to_load = [pos_idx]
                ents_to_states = []
                if (prev_context_neg != prev_context_pos or pos_next_sent != neg_next_sent) and sent_idx < neg_valid_breakpoint:
                    # if the negative is deviates from the positive, but is still valid
                    story_idxs_to_load = [pos_idx, 1-pos_idx]
                
                states_dict_list = []
                ents_to_states_list = []
                for story_idx in story_idxs_to_load:
                    # current state of entity -> {attribute -> value}
                    ents_to_states = {}
                    states_dict = {}
                    nlfied_state = []
                    precondition_state = []
                    effect_state = []
                    # ['example_id', 'base_id', 'sentences', 'entity', 'attributes', 'preconditions', 'effects', 'conflict_span', 'conflict_span_onehot', 'plausible', 'span_labels']
                    for state_type in ['preconditions', 'effects']:  #, 'attributes']:
                        if state_type == 'preconditions': state_idx = sent_idx + 1
                        else: state_idx = sent_idx
                        ents_to_states[f'all_{state_type}_facts'] = {}
                        states_dict[f'all_{state_type}_facts'] = []
                        for entity_state in example['stories'][story_idx]['entities']:
                            entity_name = entity_state['entity']
                            is_actor = entity_name == example['stories'][story_idx]['actor']
                            states_dict[f'all_{state_type}_facts'].extend(convert_attr_vector_to_nl(entity_name, is_actor, entity_state[state_type][state_idx]))
                            # ents_to_states[f'all_{state_type}_facts'][entity_name] = convert_attr_vector_to_nl(entity_name, is_actor, entity_state[state_type][state_idx])
                            ents_to_states[f'all_{state_type}_facts'][entity_name] = convert_attr_vector_to_dict(entity_state[state_type][state_idx])  # keep in numerical form
                    states_dict[f'all_preconditions_effects_facts'] = list(set(states_dict['all_preconditions_facts'] + states_dict['all_effects_facts']))
                    ents_to_states[f'all_preconditions_effects_facts'] = {entity_name: {**ents_to_states['all_preconditions_facts'][entity_name], **ents_to_states['all_effects_facts'][entity_name]} for entity_name in ents_to_states['all_effects_facts']}
                    states_dict[f'all_relevant_state_facts'] = []
                    ents_to_states['all_relevant_state_facts'] = {}
                    for attr_name in example['stories'][story_idx]['states'][sent_idx]:
                        for entity_value in example['stories'][story_idx]['states'][sent_idx][attr_name]:
                            attr_value = entity_value[1]
                            entity_name = entity_value[0]
                            if entity_name not in ents_to_states['all_relevant_state_facts']:
                                # ents_to_states['all_relevant_state_facts'][entity_name] = []
                                ents_to_states['all_relevant_state_facts'][entity_name] = {}
                            ents_to_states['all_relevant_state_facts'][entity_name][attr_name] = attr_value  # keep in numerical form
                            if attr_value == 0: continue
                            is_actor = entity_name == example['stories'][story_idx]['actor']
                            ent_fact = attribute_value_to_nl(entity_name, attr_name, attr_value, is_actor)
                            states_dict['all_relevant_state_facts'].append(ent_fact)
                            # ents_to_states['all_relevant_state_facts'][entity_name].append(ent_fact)

                    states_dict[f'all_gpt3_state_facts'] = self.facts[example['example_id']][str(sent_idx)]["pred_facts_supervision"].strip().split(". ")
                    if len(self.facts[example['example_id']][str(sent_idx)]["pred_facts_supervision"]) == 0:
                        states_dict[f'all_gpt3_state_facts'] = None
                    ents_to_states['all_gpt3_state_facts'] = {}

                    if num_aligned < max_num_aligned:
                        for state_key in ['relevant_state_facts', 'preconditions_facts', 'effects_facts',  'preconditions_effects_facts', 'gpt3_state_facts']:
                            states_dict[state_key] = states_dict[f'all_{state_key}']
                            ents_to_states[state_key] = ents_to_states[f'all_{state_key}']
                        num_aligned += 1
                    else:
                        for state_key in ['relevant_state_facts', 'preconditions_facts', 'effects_facts',  'preconditions_effects_facts', 'gpt3_state_facts']:
                            states_dict[state_key] = None
                            ents_to_states[state_key] = None
                    states_dict_list.append(states_dict)
                    ents_to_states_list.append(ents_to_states)

                if self.state_key and (self.train_state[0] == "only_fact" or self.train_state[0] == "concat_fact" or self.train_state[0] == "lang_to"):
                    any_missing_state = False
                    for state_dict in states_dict_list:
                        if state_dict[self.state_key] is None:
                            any_missing_state = True
                            break
                    if any_missing_state:
                        continue
                        # ???
                if self.classify:
                    if prev_context_pos == prev_context_neg and pos_next_sent == neg_next_sent:
                        data_entry = {
                            'context': ' '.join(prev_context_pos),
                            'next_instrs': pos_next_sent,
                            'label': 'OK',
                            # 'post_context': ' '.join(post_context_pos),
                            'states': states_dict_list[0],
                            'example_id': example['stories'][pos_idx]["example_id"],
                        }
                        all_data_processed.append(data_entry)
                    else:
                        data_entry = {
                            'context': ' '.join(prev_context_pos),
                            'next_instrs': pos_next_sent,
                            'label': 'OK',
                            # 'post_context': ' '.join(post_context_pos),
                            'states': states_dict_list[0],
                            'example_id': example['stories'][pos_idx]["example_id"],
                        }
                        all_data_processed.append(data_entry)
                        neg_label = sent_idx + 1 >= neg_valid_breakpoint
                        data_entry = {
                            'context': ' '.join(prev_context_neg),
                            'next_instrs': neg_next_sent,
                            'label': 'Not OK' if neg_label else 'OK',
                            # 'post_context': ' '.join(post_context_neg),
                            'states': states_dict_list[1] if len(states_dict_list) > 1 else {self.state_key: None},
                            'example_id': example['stories'][1-pos_idx]["example_id"],
                        }
                        all_data_processed.append(data_entry)
                else:
                    data_entry = {
                        'pos': {
                            'context': ' '.join(prev_context_pos),
                            'next_instrs': pos_next_sent,
                            'post_context': ' '.join(post_context_pos),
                            'states': states_dict_list[0],
                            'states_by_entity': ents_to_states_list[0],
                        },
                        'neg': {
                            'context': ' '.join(prev_context_neg),
                            'next_instrs': neg_next_sent,
                            'post_context': ' '.join(post_context_neg),
                            'states': states_dict_list[1] if len(states_dict_list) > 1 else {},
                            'states_by_entity': ents_to_states_list[1] if len(ents_to_states_list) > 1 else {},
                        },
                        'example_id': example['example_id'],
                    }
                    all_data_processed.append(data_entry)
        return all_data_processed
    
    def set_expected_states(self, expected_states):
        assert len(expected_states) == len(self)
        for eidx, entry in enumerate(self.data):
            self.data[eidx]['pos']['states'][self.state_key+'_expected'] = expected_states[eidx].split('. ')


class TRIPDataLoader(DataLoader):
    """
    TODO separate out entity-wise dataset/dataloader from full
    """
    def __init__(self, dataset: TRIPDataset, tokenizer, batch_size, train_state=None, max_gt_grounded_states=float("inf"), contrastive: str="False"):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        # self.dataset = dataset
        self.train_state = train_state
        self.state_key = None
        if self.train_state:
            self.state_key = train_state[-1]
        self.max_gt_grounded_states = max_gt_grounded_states
        self.contrastive = 'contrastive' in contrastive
    
    def collate_fn(self, batch):
        # get contexts and target langs
        if self.state_key is not None and 'entity' in self.state_key and not self.contrastive:
            # 'entity' in self.state_key
            batch_ctxts = []
            batch_tgts = []
        elif self.contrastive:
            batch_ctxts = []
            batch_tgts = []
            batch_labels = []
            batch_labels = []
            # (bs x n_cands[=2], seqlen)
            for i, ex in enumerate(batch):
                cand_ctxts = [ex['pos']['context'], ex['neg']['context']]
                cand_actions = [ex['pos']['next_instrs'], ex['neg']['next_instrs']]
                batch_ctxts.extend(cand_ctxts)
                batch_tgts.extend(cand_actions)
                batch_labels.extend([1, 0])
        elif self.dataset.classify:
            # if self.train_state[0] == ""
            batch_ctxts = [ex['context'] + " [Next] " + ex['next_instrs'] for ex in batch]
            batch_tgts = [ex['label'] for ex in batch]
        else:
            batch_ctxts = []
            batch_tgts = []
            for ex in batch:
                batch_ctxts.append(ex['pos']['context'])
                batch_tgts.append(ex['pos']['next_instrs'])
        
        # get states
        batch_tgt_state_ret = {}
        batch_entities = []
        batch_tgt_facts = []
        batch_all_tgt_facts = []
        batch_concat_text = []
        if self.state_key:
            if self.contrastive:
                # non-entity-wise, contrastive
                for state_story_key in [self.state_key, self.state_key+'_input', self.state_key+'_gold', self.state_key+'_concat_text']:
                    batch_tgt_state_ret[state_story_key] = []
                # (bs x n_cands[=2], seqlen)
                for i, ex in enumerate(batch):
                    assert ex['pos']['states'].get(f"all_{self.state_key}", None) is not None
                    pos_states_gold = ex['pos']['states'][f"all_{self.state_key}"]
                    pos_states = ex['pos']['states'][self.state_key]
                    if pos_states is None:
                        pos_states = []
                    if ex['neg']['states'].get(f"all_{self.state_key}", None) is not None:
                        neg_states_gold = ex['neg']['states'][f"all_{self.state_key}"]
                    else:
                        # didn't originally load because same as positive
                        try:
                            assert ex['pos']['context'] == ex['neg']['context']
                        except:
                            breakpoint()
                        neg_states_gold = pos_states_gold
                    if ex['neg']['states'].get(self.state_key, None) is not None:
                        neg_states = ex['neg']['states'][self.state_key]
                    else:
                        neg_states = pos_states
                    cand_states = ['. '.join([str(state) for state in pos_states]), '. '.join([str(state) for state in neg_states])]
                    batch_tgt_state_ret[self.state_key].extend(cand_states)
                    if self.state_key == 'preconditions':
                        cand_states_input = ['. '.join([str(state) for state in pos_states]), '. '.join([str(state) for state in pos_states])]
                    else:
                        cand_states_input = cand_states
                    batch_tgt_state_ret[self.state_key+'_input'].extend(cand_states_input)
                    batch_tgt_state_ret[self.state_key+'_gold'].extend([
                        '. '.join([str(state) for state in pos_states_gold]),
                        '. '.join([str(state) for state in neg_states_gold]),
                    ])
                    batch_tgt_state_ret[self.state_key+'_concat_text'].extend([
                        f"{ex['pos']['context']} [SEP] {cand_states[0]}",
                        f"{ex['neg']['context']} [SEP] {cand_states[1]}",
                    ])
                for state_story_key in [self.state_key, self.state_key+'_input', self.state_key+'_gold', self.state_key+'_concat_text']:
                    batch_tgt_state_ret[state_story_key] = self.tokenizer(batch_tgt_state_ret[state_story_key], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            elif self.dataset.classify:
                state_story_key = self.state_key.replace('_entity', '')
                batch_tgt_facts = []
                batch_all_tgt_facts = []
                batch_concat_text = []
                for i, ex in enumerate(batch):
                    if ex['states'].get(self.state_key, None) is not None:
                        batch_tgt_facts.append('. '.join([
                            str(state) for state in ex['states'][self.state_key]
                        ]) + " [Next] " + ex['next_instrs'])
                    else:
                        batch_tgt_facts.append('')
                    if ex['states'].get(f'all_{self.state_key}', None) is not None:
                        batch_all_tgt_facts.append('. '.join([
                            str(state) for state in ex['states'][f'all_{self.state_key}']
                        ]) + " [Next] " + ex['next_instrs'])
                    else:
                        batch_all_tgt_facts.append('')
                    if len(batch_tgt_facts) > 0:
                        batch_concat_text.append(f'{ex["context"]} [SEP] {batch_tgt_facts[-1]} [Next] {ex["next_instrs"]}')
                    else:
                        batch_concat_text.append(ex['context'])
                batch_tgt_state_ret[state_story_key] = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                batch_tgt_state_ret[state_story_key + '_input'] = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                batch_tgt_state_ret[state_story_key + '_gold'] = self.tokenizer(batch_all_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                batch_tgt_state_ret[state_story_key + '_concat_text'] = self.tokenizer(batch_concat_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            else:
                # """
                # non-entity-wise
                for story_type in ['pos', 'neg']:
                    # Include only positive states in data
                    batch_tgt_facts = [
                        '. '.join([
                            str(state) for state in ex[story_type]['states'][self.state_key]
                        ]) if ex[story_type]['states'].get(self.state_key, None) is not None else '' for ex in batch
                    ]
                    batch_expected_tgt_facts = [
                        '. '.join([
                            str(state) for state in ex[story_type]['states'][f'{self.state_key}_expected']
                        ]) if ex[story_type]['states'].get(f'{self.state_key}_expected', None) is not None else '' for ex in batch
                    ]
                    batch_all_tgt_facts = [
                        '. '.join([
                            str(state) for state in ex[story_type]['states'][f'all_{self.state_key}']
                        ]) if ex[story_type]['states'].get(f'all_{self.state_key}', None) is not None else '' for ex in batch
                    ]
                    batch_concat_text = [
                        f'{ex[story_type]["context"]} [SEP] {batch_tgt_facts[i]}' if len(batch_tgt_facts) > 0 else ex[story_type]['context'] for i, ex in enumerate(batch)
                    ]
                    if story_type == 'neg':
                        state_story_key = self.state_key + '_' + story_type
                    else:
                        state_story_key = self.state_key
                    batch_tgt_state_ret[state_story_key] = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    batch_tgt_state_ret[state_story_key + '_expected'] = self.tokenizer(batch_expected_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    batch_tgt_state_ret[state_story_key + '_input'] = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    batch_tgt_state_ret[state_story_key + '_gold'] = self.tokenizer(batch_all_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                    batch_tgt_state_ret[state_story_key + '_concat_text'] = self.tokenizer(batch_concat_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                # breakpoint()
                # """
        try:
            batch_ids = [ex['example_id'] for ex in batch]
        except:
            breakpoint()
        batch_ctxts = self.tokenizer(batch_ctxts, return_tensors='pt', padding=True, truncation=True, return_offsets_mapping=True).to(DEVICE)
        batch_tgts = self.tokenizer(batch_tgts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        if self.contrastive:
            batch_tgts['labels'] = torch.tensor(batch_labels).long().to(DEVICE)
            batch_tgts['num_cands'] = 2
        # (inputs, lang_tgts, init_state, tgt_state, game_ids, entities)
        return batch_ctxts, batch_tgts, None, batch_tgt_state_ret, batch_ids, batch_entities

    def evaluate(self, model, inputs, lang_tgts, states=None):
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        if self.train_state:
            if self.train_state[0] == "only_fact":
                inputs = states[self.state_key+'_input']
            elif self.train_state[0] == "concat_fact":
                inputs = states[self.state_key+'_concat_text']
            elif self.train_state[0] == "lang_to":
                lang_tgts = states[self.state_key]

        if self.contrastive:
            # (bs x n_cands, seqlen)
            if type(model) == ContrastiveClassifierHead:
                # (bs x n_cands, seqlen)
                return_dict = model.loss(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['labels'],
                    raw_cand_outs={'input_ids': cand_input_ids, 'attention_mask': cand_attn_mask}, return_dict=True,
                )
                batch_lang_loss = return_dict['similarity']
            else:
                # (bs x n_cands, seqlen)
                return_dict = model(
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=lang_tgts['input_ids'], return_dict=True,
                )
                # (bs, n_cands, seqlen)
                input_input_ids = inputs['input_ids'].view(-1, lang_tgts['num_cands'], inputs['input_ids'].size(-1))
                # (bs, n_cands, seqlen)
                tgt_input_ids = lang_tgts['input_ids'].view(-1, lang_tgts['num_cands'], lang_tgts['input_ids'].size(-1))
                if self.state_key:
                    # (bs, n_cands, seqlen)
                    state_input_ids = states[self.state_key]['input_ids'].view(-1, lang_tgts['num_cands'], states[self.state_key]['input_ids'].size(-1))
                # (bs, n_cands, seqlen, vocab_size)
                pred_logits = return_dict.logits.view(-1, lang_tgts['num_cands'], return_dict.logits.size(-2), return_dict.logits.size(-1))
                pred_logits = pred_logits.permute(0,3,1,2)
                batch_lang_loss = loss_fct(pred_logits, tgt_input_ids)
                # (bs, n_cands, seqlen) -> (bs, n_cands)
                batch_lang_loss = batch_lang_loss.sum(-1) / (tgt_input_ids != self.tokenizer.pad_token_id).sum(-1)
                labels = lang_tgts['labels'].view(-1, lang_tgts['num_cands'])
            lang_loss = return_dict['loss']
            # lowest scoring losses per example in batch
            # (bs,)
            min_values, chosen_tgt = batch_lang_loss.min(-1)
            # # randomize if multiple minimal values
            # breakpoint()
            # mask with `True` at entries which are minimal values for that row
            row_min_value_mask = batch_lang_loss == min_values.unsqueeze(-1)
            # for row in 
            # # if row has multiple minimal values, choose randomly
            # row_indices = torch.arange(row_min_value_mask.size(-1)).unsqueeze(0).repeat(row_min_value_mask.size(0),1).to(row_min_value_mask.device)
            # row_min_value_mask.sum(-1)
            # row_indices[row_min_value_mask]
            #     row_min_value_mask.sum(-1) > 1][row_min_value_mask[]]

            actual_tgt = (labels == 1).nonzero(as_tuple=False)[:,1]

            bs = actual_tgt.size(0)
            n_correct = 0
            batch_save_preds = []
            for idx in range(bs):
                if row_min_value_mask[idx].sum() > 1:
                    chosen_tgt[idx] = random.choice(row_min_value_mask[idx].nonzero(as_tuple=False))
                correct = chosen_tgt[idx] == actual_tgt[idx]
                n_correct += correct
                batch_save_preds.append({
                    'pos_ctxt': self.tokenizer.decode(input_input_ids[idx,actual_tgt[idx]], skip_special_tokens=True),
                    'pos_tgt': self.tokenizer.decode(tgt_input_ids[idx,actual_tgt[idx]], skip_special_tokens=True),
                    'neg_ctxts': [self.tokenizer.decode(input_input_ids[idx,tgt_idx], skip_special_tokens=True) for tgt_idx in range(labels.size(-1)) if tgt_idx != actual_tgt[idx]],
                    'neg_tgts': [self.tokenizer.decode(tgt_input_ids[idx,tgt_idx], skip_special_tokens=True) for tgt_idx in range(labels.size(-1)) if tgt_idx != actual_tgt[idx]],
                    'correct?': correct.item(),
                    'scores': {
                        'pos': batch_lang_loss[idx,actual_tgt[idx]].item(),
                        'neg': [batch_lang_loss[idx,tgt_idx].item() for tgt_idx in range(labels.size(-1)) if tgt_idx != actual_tgt[idx]],
                    },
                })
                if not correct:
                    if chosen_tgt[idx] > actual_tgt[idx]:
                        batch_save_preds[-1]['neg_pred_idx'] = chosen_tgt[idx].item() - 1
                    else:
                        batch_save_preds[-1]['neg_pred_idx'] = chosen_tgt[idx].item()
                # add state
                if self.state_key:
                    batch_save_preds[-1]['pos_states'] = self.tokenizer.decode(state_input_ids[idx,actual_tgt[idx]], skip_special_tokens=True)
                    batch_save_preds[-1]['neg_states'] = [self.tokenizer.decode(state_input_ids[idx,tgt_idx], skip_special_tokens=True) for tgt_idx in range(labels.size(-1)) if tgt_idx != actual_tgt[idx]]
            # # (bs, 1(#samples), seqlen)
            # generated_next_utt = torch.stack([tgt_input_ids[tgt_idx, chosen_tgt[tgt_idx]] for tgt_idx in range(chosen_tgt.size(0))]).unsqueeze(1)
            # # (bs x n_cands,)
            # chosen_tgt = generated_next_utt.repeat_interleave(2, dim=0)
        else:
            breakpoint()
            encoder_outputs: ModelOutput = encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
            # assume bart
            decoder_inputs = {
                'input_ids': None,
                'attention_mask': inputs['attention_mask'],
                'encoder_outputs': encoder_outputs,
                'labels': lang_tgts['input_ids'],  # automatically generates `decoder_input_ids` out of labels
                'return_dict': True,
            }
            return_dict = model(**decoder_inputs)
            # return_dict = model(input_ids=None, encoder_outputs=encoder_outputs, decoder_input_ids=lang_tgts['input_ids'], labels=lang_tgts['input_ids'], return_dict=True,)
            lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
            generated_next_utt = None
        return lang_loss, n_correct, batch_save_preds, bs
    
    def select_vectors_corresponding_to_entity(self, encoder_outputs, inputs, entities):
        """
        Select vectors of `encoder_outputs` that correspond to tokens of `entities` in `inputs`
        """
        bs = encoder_outputs[0].size(0)
        selected_encoder_outputs = []
        for i in range(bs):
            nl_input = self.tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
            offset_mapping = inputs['offset_mapping'][i][inputs['attention_mask'][i].bool()]
            if self.tokenizer.eos_token:
                # remove EOS token
                offset_mapping = offset_mapping[:-1,:]
            idxs_of_entity_tokens = []
            # find all offsets
            for m in re.finditer(f" {entities[i]}", f" {nl_input}"):
                span_start = max(m.span()[0] - 1, 0)
                span_end = m.span()[1] - 1
                # largest index <= start of span
                entity_offset_start_idx = (offset_mapping[:,0] <= span_start).nonzero(as_tuple=False).max()
                # smallest index >= end of span
                entity_offset_end_idx = (offset_mapping[:,1] >= span_end).nonzero(as_tuple=False).min()
                idxs_of_entity_tokens.extend(list(range(entity_offset_start_idx, entity_offset_end_idx+1)))
                assert entities[i] in self.tokenizer.decode(inputs['input_ids'][i,entity_offset_start_idx:entity_offset_end_idx+1])
                # if (entity_offset_start_mask & entity_offset_end_mask).any():
                #     entity_offset_idx = (entity_offset_start_mask & entity_offset_end_mask).nonzero(as_tuple=False).squeeze().item()
                #     idxs_of_entity_tokens.append(entity_offset_idx)
                # else:
                #     try:
                #         entity_offset_start_idx = entity_offset_start_mask.nonzero(as_tuple=False).squeeze().item()
                #     except:
                #         breakpoint()
                #     entity_offset_end_idx = entity_offset_end_mask.nonzero(as_tuple=False).squeeze().item()
            if len(idxs_of_entity_tokens) == 0:
                # entity not found...
                selected_encoder_outputs.append(encoder_outputs[0][i].mean(0))
            else:
                idxs_of_entity_tokens = torch.tensor(idxs_of_entity_tokens).to(offset_mapping.device)
                try:
                    selected_encoder_outputs.append(encoder_outputs[0][i][idxs_of_entity_tokens].mean(0))
                except:
                    breakpoint()
        # average(?) over encodings
        selected_encoder_outputs = torch.stack(selected_encoder_outputs)
        
        return (selected_encoder_outputs,)


get_trip_consistency = None