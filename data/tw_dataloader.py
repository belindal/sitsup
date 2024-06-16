import torch
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader, Dataset, IterableDataset
from utils import DEVICE
import glob
from data.parse_tw import (
    parse_facts_to_nl, EntitySet, ENTITIES_SIMPLE, ROOMS_SIMPLE,
    control_pairs_simple, control_pairs_with_rooms_simple,
    gen_possible_pairs, gen_negative_tgts,
    control_tgt_to_mention_simple, control_tgt_to_mention_with_rooms_simple,
    pad_stack, get_relevant_facts_about, remap_entset, convert_fileid_to_gameid,
)
import itertools
from transformers import PreTrainedTokenizerBase
import regex as re
import random


class TWDatasetGPT3(Dataset):
    def __init__(self, data_dir, data_split, start_idx=0, end_idx=-1, tokenizer=None, inform7_game=None, debug=False):
        self.data_dir = data_dir
        self.data_split = data_split
        self.label2stories = {}
        self.data_order_file = os.path.join(data_dir, f"{data_split}_data_order.json")
        self.tokenizer = tokenizer
        if os.path.exists(self.data_order_file):
            self.data_order = json.load(open(self.data_order_file))
        else:
            self.data_order = None
        self.inform7_game = inform7_game
        
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_seq_len = 2048
        self.debug = debug

        # build data
        self.data = self.load_data(data_dir, data_split)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]

    def load_data(self, data_dir, data_split):
        all_data_processed = []
        # {'contexts': [], 'post_contexts': [], 'tgts': [], 'final_states': [], 'init_states': [], 'filenames': []}  # init state + actions
        n_states = 0
        files = sorted(glob.glob(os.path.join(os.path.join(data_dir, data_split), "*_states.txt")))
        if not self.data_order:
            random.seed(0)
            random.shuffle(files)
        all_ids = []
        for fp in tqdm(files):
            if self.debug and len(all_data_processed) > 50: break
            all_actions = []  # actions that make up current file
            curr_action = []  # lines that make up current action
            n_cutoff_actions = 0  # num actions for max_seq_len (not strictly necessary, just ensures we don't run for too long)
            states = []
            valid_actions = []
            invalid_actions = []
            # create all_actions (file, separated by commands, aka '>')
            langs_file = fp.replace('_states.txt', '.txt')
            with open(langs_file) as f:
                approx_num_toks = 0
                for ln, line in enumerate(f):
                    if (line.strip().startswith("***") and line.strip().endswith("***")) or approx_num_toks > 2*self.max_seq_len:
                        # loop will always end on this condition, since "The End" is in all documents
                        break
                    # line = line.strip() + ' | '
                    if ln == 0: continue
                    if line.startswith(">"):
                        action = ''.join(curr_action)
                        if approx_num_toks <= self.max_seq_len: n_cutoff_actions += 1
                        all_actions.append(action)
                        curr_action = []
                    curr_action.append(line)
                    approx_num_toks += len(self.tokenizer.tokenize(line))
                # get last part
                if line.startswith(">") and approx_num_toks + len(self.tokenizer.tokenize(line)) <= self.max_seq_len:
                    all_actions.append(''.join(curr_action))
                    if approx_num_toks + len(self.tokenizer.tokenize(line)) <= self.max_seq_len: n_cutoff_actions += 1
            # create final_states
            with open(fp) as f:
                num_lines = 0
                for line in f:
                    if num_lines > n_cutoff_actions + 1:  #+1 for initial state
                        break
                    state = json.loads(line)
                    # breakpoint()
                    true_facts = parse_facts_to_nl(state['curr_state_belief_facts']['true'], self.inform7_game)
                    false_facts = parse_facts_to_nl(state['curr_state_belief_facts']['false'], self.inform7_game)
                    states.append(true_facts)
                    valid_actions.append(['> ' + va.split('\n')[0] for va in state['valid_actions']])
                    invalid_actions.append(['> ' + iva.split('\n')[0] for iva in state['invalid_actions']])
                    num_lines += 1

            if n_cutoff_actions <= 2: continue
            # breakpoint()
            # assert len(states) == len(context_sentences)
            # create (context, next utterance, init_state, states) tuples for each dataset from all_actions
            # (all_actions[0], all_actions[1], states[0], states[0]);
            # (all_actions[0:1], all_actions[2], states[0], states[1]);
            # (all_actions[0:2], all_actions[3], states[0], states[2]);
            # ...
            # NOTE states[i] is state *after* `i`th action, so use (i-1) to get state immediately after context (actions 1...i-1)
            interacted_entities = set()
            # s = 1  # after all_actions[0]
            context_sentences = [''.join([all_actions[0], all_actions[1]])]
            all_valid_actions_list = []
            all_invalid_actions_list = []
            actions_list = []
            states_list = []
            world = os.path.split(langs_file)
            world = os.path.join(os.path.split(world[0])[1], world[1])
            for c in range(2,n_cutoff_actions):
                # prev_actions = ''.join(all_actions[1:c])
                # tgt_action = all_actions[c].split('[')[0]
                postfix = ''.join(all_actions[c:])
                assert len(postfix) != 0
                # contexts.append(curr_context)
                context_sentences.append(all_actions[c])
                
                states_list.append(states[c-1])
                # actions_list.append(tgt_action)
                all_valid_actions_list.append([va for va in valid_actions[c-1] if va not in ["> look", "> inventory"] and not va.startswith("> examine")])
                all_invalid_actions_list.append([iva for iva in invalid_actions[c-1] if not iva.startswith("> unlock")])
                # all_valid_actions_list.append([va for va in valid_actions[c-1] if va not in ["> look", "> inventory"] and not va.startswith("> examine")])
                # all_invalid_actions_list.append([iva for iva in invalid_actions[c-1] if not iva.startswith("> unlock")])
                # increment_corresponding_state = all_actions[c-1].startswith(">")  # last action in context
                # if increment_corresponding_state:
                #     s += 1
                #     n_states += 1
            assert len(context_sentences) > 0
            entry = {
                "input_sents": context_sentences,
                # "post_contexts": postfix,
                # "tgts": tgt_action,
                # "init_states": states[0],
                # "final_states": states[s],
                "filenames": world,
                "id": world,
                # "actions": actions_list,
                "states": states_list,
                "all_valid_actions": all_valid_actions_list,
                "all_invalid_actions": all_invalid_actions_list,
            }
            all_data_processed.append(entry)
            all_ids.append(world)

        if self.data_order is None:
            all_ids = list(set(all_ids))
            random.seed(0)
            random.shuffle(all_ids)
            self.data_order = all_ids
            json.dump(all_ids, open(self.data_order_file, "w"))
        all_data_processed.sort(key=lambda x: self.data_order.index(x['filenames']))
        if self.end_idx > -1:
            all_data_processed = all_data_processed[:self.end_idx]
        all_data_processed = all_data_processed[self.start_idx:]
        return all_data_processed


class TWDataset(Dataset):
    def __init__(
        self, data_dir, tokenizer, data_split, max_seq_len=512,
        max_data_size=10000, interleave_state_in_ctxt=False, pred_action_and_response_joint=False,
        inform7_game=None, randseed=None, logger=None, *args, **kwargs,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_split = data_split
        self.max_seq_len = max_seq_len
        self.max_data_size = max_data_size
        self.interleave_state_in_ctxt = interleave_state_in_ctxt
        self.pred_action_and_response_joint = pred_action_and_response_joint
        self.inform7_game = inform7_game
        self.randseed = randseed
        self.logger = logger

        # build data
        self.load_data()
    
    def __len__(self):
        return len(self.data['contexts'])
    
    def __getitem__(self, i):
        item = {
            k: self.data[k][i] for k in self.data
        }
        item['data_idx'] = i
        return item
    
    def get_gameids(self):
        game_ids = []
        game_ids_set = set()
        # uniquify
        for fn in self.data['filenames']:
            game_id = fn.split('_')[0] 
            if game_id not in game_ids_set:
                game_ids.append(game_id)
                game_ids_set.add(game_id)
        return game_ids

    def load_data(self):
        init_actions_data = {'contexts': [], 'post_contexts': [], 'tgts': [], 'final_states': [], 'init_states': [], 'filenames': []}  # init state + actions
        n_states = 0
        files = sorted(glob.glob(os.path.join(os.path.join(self.data_dir, self.data_split), "*_states.txt")))
        if self.randseed:
            random.seed(self.randseed)
            random.shuffle(files)
        for fp in tqdm(files):
            all_actions = []  # actions that make up current file
            curr_action = []  # lines that make up current action
            n_cutoff_actions = 0  # num actions for max_seq_len (not strictly necessary, just ensures we don't run for too long)
            states = []
            # create all_actions (file, separated by commands, aka '>')
            langs_file = fp.replace('_states.txt', '.txt')
            with open(langs_file) as f:
                approx_num_toks = 0
                for line in f:
                    if (line.strip().startswith("***") and line.strip().endswith("***")) or approx_num_toks > 2*self.max_seq_len:
                        # loop will always end on this condition, since "The End" is in all documents
                        break
                    line = line.strip() + ' | '
                    if line.startswith(">"):
                        action = ''.join(curr_action)
                        if approx_num_toks <= self.max_seq_len: n_cutoff_actions += 1
                        all_actions.append(action)
                        curr_action = []
                    curr_action.append(line)
                    approx_num_toks += len(self.tokenizer.tokenize(line))
                    if not self.pred_action_and_response_joint and line.startswith(">"):
                        # if action, add line immediately
                        action = ''.join(curr_action)
                        if approx_num_toks <= self.max_seq_len: n_cutoff_actions += 1
                        all_actions.append(action)
                        curr_action = []
                # get last part
                if line.startswith(">") and approx_num_toks + len(self.tokenizer.tokenize(line)) <= 2*self.max_seq_len:
                    all_actions.append(''.join(curr_action))
                    if approx_num_toks + len(self.tokenizer.tokenize(line)) <= self.max_seq_len: n_cutoff_actions += 1
            # create final_states
            with open(fp) as f:
                num_lines = 0
                for line in f:
                    if num_lines > n_cutoff_actions + 1:  #+1 for initial state
                        break
                    state = json.loads(line)
                    states.append(state)
                    num_lines += 1

            if self.interleave_state_in_ctxt:
                all_actions = [
                    f"{all_actions[c]}[{'. '.join(parse_facts_to_nl(states[c]['added_belief_facts']['true'], self.inform7_game))}] ### "
                    for c in range(n_cutoff_actions)
                ]
                
            # create (context, next utterance, init_state, states) tuples for each dataset from all_actions
            # (all_actions[0], all_actions[1], states[0], states[0]);
            # (all_actions[0:1], all_actions[2], states[0], states[1]);
            # (all_actions[0:2], all_actions[3], states[0], states[2]);
            # ...
            # NOTE states[i] is state *after* `i`th action, so use (i-1) to get state immediately after context (actions 1...i-1)
            interacted_entities = set()
            s = 0  # after all_actions[0]
            for c in range(2,n_cutoff_actions):
                world = os.path.split(langs_file)
                world = os.path.join(os.path.split(world[0])[1], world[1])
                actions = ''.join(all_actions[1:c])
                tgt_action = all_actions[c].split('[')[0]
                postfix = ''.join(all_actions[c:])
                assert len(postfix) != 0
                increment_corresponding_state = all_actions[c-1].startswith(">")  # last action in context
                if increment_corresponding_state:
                    s += 1
                    n_states += 1

                goal = all_actions[0].split(' | ')[0]
                curr_context = ''.join([all_actions[0].replace(goal, ""), actions])
                init_actions_data['contexts'].append(curr_context)
                init_actions_data['post_contexts'].append(postfix)
                init_actions_data['tgts'].append(tgt_action)
                init_actions_data['init_states'].append(states[0])
                init_actions_data['final_states'].append(states[s])
                init_actions_data['filenames'].append(world)

                if len(init_actions_data['contexts']) >= self.max_data_size:
                    break
            if len(init_actions_data['contexts']) >= self.max_data_size:
                break
        for k in init_actions_data:
            assert len(init_actions_data[k]) == len(init_actions_data['contexts'])
        if self.logger: self.logger.info(f"Using files order: {init_actions_data['filenames']}")
        self.data = init_actions_data


class TWEntitySetDataset(TWDataset):
    """
    Same context for each entity pair, and return facts for each entity pair
    """
    def __init__(
        self, data_dir, tokenizer, data_split,
        ent_set_size, control, gamefile, state_key, tgt_state_key='final_states',
        max_seq_len=512, max_data_size=10000, interleave_state_in_ctxt=False,
        pred_action_and_response_joint=False, inform7_game=None, randseed=None, control_input=False,
        possible_pairs=None, precomputed_negs=None,
    ):
        """
        control_input: show same input
        """
        self.gamefile = gamefile
        self.inform7_game = inform7_game
        if 'simple' in self.gamefile:
            self.all_entities = ENTITIES_SIMPLE + ROOMS_SIMPLE
            self.control_tgt_to_mention = control_tgt_to_mention_simple
            self.control_tgt_to_mention_with_rooms = control_tgt_to_mention_with_rooms_simple
            self.all_rooms = ROOMS_SIMPLE
        # elif 'treasure_hunter' in self.gamefile: TODO outdated ENTITIES_TH
        #     self.all_entities = ENTITIES_TH.union(ROOMS_TH)
        #     self.control_tgt_to_mention = control_tgt_to_mention_TH
        #     self.control_tgt_to_mention_with_rooms = control_tgt_to_mention_with_rooms_TH
        #     self.all_rooms = ROOMS_TH
        self.tgt_state_key = tgt_state_key
        self.ent_set_size = ent_set_size
        self.control = control
        self.state_key = state_key

        self.possible_pairs = possible_pairs
        self.precomputed_negs = precomputed_negs
        super().__init__(
            data_dir=data_dir, tokenizer=tokenizer, data_split=data_split,
            max_seq_len=max_seq_len, max_data_size=max_data_size, 
            interleave_state_in_ctxt=interleave_state_in_ctxt, pred_action_and_response_joint=pred_action_and_response_joint,
            inform7_game=inform7_game, randseed=randseed,
        )

    def find_mention_in_ctxt(self, entity, ctxt):
        """
        check which form of entity is mentioned in the context
        If not found, returns `None`
        """
        # candidates = [f'-= {entity.title()} =-'] + [
        #     f'{prefix}{ent_case}{suffix}' for prefix in [' ', '\n'] for ent_case in [entity, entity.title()] for suffix in [' ', '\n', ',', '.', '!', '?']
        # ]
        candidates = [f'[ |\n][{entity[0].lower()}|{entity[0].upper()}]{entity[1:].lower()}[ |\n|,|\.|!|\?]']
        candidates.append(f'-= {entity.title()} =-')
        # re.findall()
        # first_mention = None
        # first_mention_location = float("inf")
        all_mention_locations = None
        for cand in candidates:
            mention_location = list(re.finditer(cand, ctxt))
            if len(mention_location) > 0:
                if all_mention_locations is None: all_mention_locations = []
                all_mention_locations += mention_location
                # if mention_location < first_mention_location:
                #     first_mention = cand
                #     first_mention_location = mention_location
        return all_mention_locations

    def load_data(self):
        super().load_data()
        # compute negs here (if not already pre-loaded from file...)
        if self.precomputed_negs is None:
            if self.ent_set_size == 2:
                self.possible_pairs = load_possible_pairs(data_dir=self.gamefile, game_ids=self.get_gameids())
            self.precomputed_negs = load_negative_tgts(data_dir=self.gamefile, tokenizer=self.tokenizer, game_ids=self.get_gameids(), ent_set_size=self.ent_set_size)

        entities_data = {
            'contexts': [], 'tgts': [], 'init_states': [], 'tgt_states': [], 'filenames': [], 'game_ids': [],
            'entities': [], 'mentions': [], 'labels': [], 'all_states_tokenized': [], 'all_states_encoded': [],
        }
        print("Computing entities")
        # getting facts works only for returning 1 type of fact...
        for i in tqdm(range(len(self.data['contexts']))):
            context = self.data['contexts'][i]
            game_id = self.data['filenames'][i].split('_')[0]
            tgt = self.data['tgts'][i]
            init_state, tgt_state = self.data['init_states'][i][self.state_key], self.data[self.tgt_state_key][i][self.state_key]
            init_state = {tf: ' [SEP] '.join(parse_facts_to_nl(init_state[tf], self.inform7_game)) for tf in init_state}
            # get all entities mentioned in context
            # + transforms their names as appropriate
            entities = []
            ent2mentions = {}  # form of entity as mentioned in text
            for e in self.all_entities:
                if e == 'P': e = 'player'
                if e == 'I': e = 'inventory'
                e_in_ctxt = self.find_mention_in_ctxt(e, context)
                if not e_in_ctxt: continue
                ent2mentions[e] = e_in_ctxt
                if self.control:
                    ce = self.control_tgt_to_mention_with_rooms[e] if e in self.control_tgt_to_mention_with_rooms else e
                    ce_in_ctxt = self.find_mention_in_ctxt(ce, context)
                    if not ce_in_ctxt: continue
                    ent2mentions[ce] = ce_in_ctxt
                entities.append(e)

            all_entities_list = list(itertools.combinations([None, *entities], self.ent_set_size))
            # create all entity pairs
            for ent_list in all_entities_list:
                entset = EntitySet(ent_list)
                if self.possible_pairs is not None and entset not in self.possible_pairs[game_id]: continue

                # get all facts for list of entities
                ent_facts = {}
                for tf in tgt_state:
                    relevant_facts = get_relevant_facts_about(entset, tgt_state[tf], None, None, exact_arg_count=(self.ent_set_size > 1), exact_arg_order=False)
                    ent_facts[tf] = relevant_facts

                if self.control == 'control':
                    entset = EntitySet(remap_entset(entset, self.control_tgt_to_mention))
                if self.control == 'control_rooms':
                    entset = EntitySet(remap_entset(entset, self.control_tgt_to_mention_with_rooms))
                mentionset = remap_entset(entset, ent2mentions)

                entities_data['contexts'].append(context)
                entities_data['tgts'].append(tgt)
                entities_data['filenames'].append(self.data['filenames'][i])
                entities_data['game_ids'].append(game_id)
                entities_data['init_states'].append(init_state)
                ent_facts = {tf: ' [SEP] '.join(parse_facts_to_nl(ent_facts[tf], self.inform7_game)) for tf in ent_facts}
                entities_data['tgt_states'].append(ent_facts)
                entities_data['entities'].append(entset)
                entities_data['mentions'].append(mentionset)

                labels, all_states_inputs, all_states_vectors = self.get_matching_state_label(entset, ent_facts, self.precomputed_negs)
                entities_data['labels'].append(labels)
                entities_data['all_states_tokenized'].append(all_states_inputs)
                entities_data['all_states_encoded'].append(all_states_vectors)
        self.data = entities_data

    def get_matching_state_label(self, entset, target_state, precomputed_negs):
        '''
        create targets
        (pos/neg/unk)
        '''
        all_input_tokens = precomputed_negs['all_entity_inputs'][EntitySet.serialize(entset)].to('cpu')
        all_vectors = precomputed_negs['all_entity_vectors'][EntitySet.serialize(entset)]
        fact_to_idx = precomputed_negs['state_to_idx'][EntitySet.serialize(entset)]
        all_inputs = precomputed_negs['idx_to_state'][EntitySet.serialize(entset)]

        '''
        create labels
        '''
        labels = [0 for _ in range(len(fact_to_idx))]
        for i, tf in enumerate(target_state):
            if len(target_state[tf]) > 0:
                for fact in target_state[tf].split(' [SEP] '):
                    if 'carries' in fact:
                        if fact not in fact_to_idx:
                            fact = fact.replace("The player carries ", "") + " is in the inventory"
                    fact = f'{fact[0].upper()}{fact[1:]}'
                    try: labels[fact_to_idx[fact]] = i + 1
                    except: import pdb; pdb.set_trace()
        return labels, all_input_tokens, all_vectors


class TWEntitySetDataLoader(DataLoader):
    def __init__(
        self, dataset: TWDataset, tokenizer: PreTrainedTokenizerBase, batch_size: int, control_input: bool = False,
    ):
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.control_input = control_input
        
    """
    state_keys_to_get: [(init/final_state, key); (init/final_state, key)]
    nnegs: # negatives to get (set 0 to not get negatives, inf to get all negatives)
    npos: # positives to get (default 1)
    expected_states: to use for EM (in cases where gold-annotated states are unavailable)
    """
    
    # TODO just 1 key to  make this code less messy...
    def tokenize_truncate(self, inputs, mask, max_len,):
        """
        tensor (bsz, seqlen, *)
        mask (bsz)
        max_len (int)
        """
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=False)
        return {k: apply_mask_and_truncate(tokenized_inputs[k], mask, max_len,) for k in tokenized_inputs}
    
    def collate_fn(self, batch):
        new_batch = {k: [] for k in batch[0]}
        for i, item in enumerate(batch):
            for k in item:
                new_batch[k].append(item[k])
        batch = new_batch
        
        # get context
        if self.control_input:
            context = ','.join([' ' + mention + ' ' for mention in batch['entities'] if mention is not None])
        else:
            context = batch['contexts']
        context_tokens = self.tokenizer(context, return_tensors='pt', padding=True, truncation=False, return_offsets_mapping=True)
        # get contexts within max length of model
        items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
        if not items_to_keep.any():
            return None, None, None, None, batch['game_ids'], None
        context_tokens = {k: apply_mask_and_truncate(context_tokens[k], items_to_keep, self.tokenizer.model_max_length) for k in context_tokens}
        # get lang tgts
        tgt_tokens = self.tokenize_truncate(batch['tgts'], items_to_keep, self.tokenizer.model_max_length)
        # get state tgts
        state_tokens = {}
        for state_type in ['init_states', 'tgt_states']:
            state_tokens[state_type] = {}
            for tf in ['true', 'false']:
                tokenized_state = self.tokenize_truncate([state[tf] for state in batch[state_type]], items_to_keep, self.tokenizer.model_max_length)
                state_token_key = f'{k}_{tf}' if tf == 'false' else k
                for k in tokenized_state: state_tokens[state_type][state_token_key] = tokenized_state[k]
        # mention sets/entity sets/gameids
        entity_sets = {
            'mentions': [ent for idx, ent in enumerate(batch['mentions']) if items_to_keep[idx]],
            'entities': [ent for idx, ent in enumerate(batch['entities']) if items_to_keep[idx]],
        }
        game_ids = [gid for idx, gid in enumerate(batch['game_ids']) if items_to_keep[idx]]

        # labels
        # (bs, # facts, seqlen[, embeddim])
        state_tokens['tgt_states']['all_states_input_ids'], state_tokens['tgt_states']['all_states_attn_mask'] = pad_stack(batch['all_states_tokenized'], pad_idx=self.tokenizer.pad_token_id, device=DEVICE)
        state_tokens['tgt_states']['all_states_encoding'], encoding_mask = pad_stack(batch['all_states_encoded'], pad_idx=0, device=DEVICE)
        assert (encoding_mask == state_tokens['tgt_states']['all_states_attn_mask']).all()
        max_nfacts = state_tokens['tgt_states']['all_states_input_ids'].size(1)
        # (bs, # facts)
        labels = [lentry + [-1 for _ in range(max_nfacts-len(lentry))] for lentry in batch['labels']]
        labels = torch.tensor(labels).to(DEVICE)
        assert ((labels != -1) == state_tokens['tgt_states']['all_states_attn_mask'][:,:,0]).all()
        # # at least 1 fact per batch
        # assert (labels.abs().sum(1) > 0).all()
        state_tokens['tgt_states']['labels'] = labels

        return context_tokens, tgt_tokens, state_tokens['init_states'], state_tokens['tgt_states'], game_ids, entity_sets


class TWFullDataLoader(DataLoader):
    def __init__(
        self, dataset: TWDataset, gamefile, tokenizer, batch_size, state_keys_to_get=[], max_gt_grounded_states=float("inf"), states=None,
        append_facts_to_input=False, include_feedback=False, nnegs=0, npos=1
    ):
        """
        states: new set of states (must have 1 per sample for entire dataset) to override loaded states
        """
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        self.gamefile = gamefile
        self.state_keys_to_get = state_keys_to_get
        self.max_gt_grounded_states = max_gt_grounded_states
        if len(state_keys_to_get) == 0:
            self.state_keys = []  # which field
            self.tgt_state_keys = []
            self.tgt_state_key = 'final_state'
        else:
            self.state_keys = [key[1].replace('_single', '').replace('_pair', '') for key in state_keys_to_get]
            self.tgt_state_keys = [key[0]+'_state' for key in state_keys_to_get]
            if len(state_keys_to_get) == 1:
                get_pair = state_keys_to_get[0][1].endswith('_pair')
                get_single = state_keys_to_get[0][1].endswith('_single')
                state_key = self.state_keys[0]
                self.tgt_state_key = self.tgt_state_keys[0]
        self.states = states
        self.nnegs = nnegs
        self.npos = npos
        self.append_facts_to_input = append_facts_to_input
    
    def update_state(self, new_states):
        self.states = new_states

    def collate_fn(self, batch):
        game_ids = [item['filenames'].split('_')[0] for item in batch]
        contexts = [item['contexts'] for item in batch]
        post_contexts = [item['post_contexts'] for item in batch]
        context_tokens = self.tokenizer(contexts, return_tensors='pt', padding=True, truncation=False)
        post_context_tokens = self.tokenizer(post_contexts, return_tensors='pt', padding=True, truncation=True, max_length=128)
        full_context_tokens = self.tokenizer([
            contexts[j] + ' [SEP] ' + post_contexts[j] for j in range(len(batch))
        ], return_tensors='pt', padding=True, truncation=False)
        items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
        if not items_to_keep.any():
            return None, None, None, None, game_ids, None

        # Delete problematic example(s) + truncate rest
        context_tokens = {key: apply_mask_and_truncate(context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in context_tokens}
        post_context_tokens = {key: apply_mask_and_truncate(post_context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in post_context_tokens}
        full_context_tokens = {key: apply_mask_and_truncate(full_context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in full_context_tokens}
        if self.nnegs > 0 or self.npos > 1:
            tgt_tokens = self.tokenizer(batch['tgts'], return_tensors='pt', padding=True, truncation=True)
            tgt_tokens['all_cands_inputs'] = [] #{'input_ids': [], 'attention_mask': []}
            tgt_tokens['labels'] = []
            tgt_tokens['valid_actions'] = []
            tgt_tokens['invalid_actions'] = []
            for j in range(len(batch)):
                tgt_tokens['valid_actions'].append(batch[j][self.tgt_state_key]['valid_actions'])
                tgt_tokens['invalid_actions'].append(batch[j][self.tgt_state_key]['invalid_actions'])
                # sample
                if training:
                    nnegs = min(self.nnegs, len(batch[j][self.tgt_state_key]['invalid_actions']))
                    npos = min(self.npos, len(batch[j][self.tgt_state_key]['valid_actions']))
                else:
                    nnegs = min(self.nnegs, len(batch[j][self.tgt_state_key]['invalid_actions']))
                    npos = min(self.npos, len(batch[j][self.tgt_state_key]['valid_actions']))
                valid_inputs = random.sample(batch[j][self.tgt_state_key]['valid_actions'], npos)
                if include_feedback: valid_inputs = ["> "+vi for vi in valid_inputs]
                else: valid_inputs = ["> "+vi.split('\n')[0] for vi in valid_inputs]
                invalid_inputs = random.sample(batch[j][self.tgt_state_key]['invalid_actions'], nnegs)
                if include_feedback: invalid_inputs = ["> "+ii for ii in invalid_inputs]
                else: invalid_inputs = ["> "+ii.split('\n')[0] for ii in invalid_inputs]
                cand_tokens = self.tokenizer(valid_inputs + invalid_inputs, return_tensors='pt', padding=True, truncation=True)
                tgt_tokens['all_cands_inputs'].append(cand_tokens)
                tgt_tokens['labels'].append([1 for _ in valid_inputs] + [0 for _ in invalid_inputs])
            # (bs, n_cands, seqlen)
            tgt_tokens['all_cands_input_ids'], tgt_tokens['all_cands_attention_mask'] = pad_stack(
                tgt_tokens['all_cands_inputs'], pad_idx=self.tokenizer.pad_token_id, device=DEVICE)
            n_cands = tgt_tokens['all_cands_input_ids'].size(1)
            # (bs, n_cands,)
            tgt_tokens['labels'] = torch.tensor([label + [0 for _ in range(n_cands-len(label))] for label in tgt_tokens['labels']]).to(DEVICE)
        else:
            # create target tokens and states
            tgts = [item['tgts'] for item in batch]
            tgt_tokens = self.tokenizer(tgts, return_tensors='pt', padding=True, truncation=True)
        # delete problem examples
        tgt_tokens = {key: apply_mask_and_truncate(tgt_tokens[key], items_to_keep, self.tokenizer.model_max_length) if type(tgt_tokens[key]) == torch.Tensor else tgt_tokens[key] for key in tgt_tokens}
        
        init_states = {}
        final_state = {}
        for sk, state_key in enumerate(self.state_keys):
            for suffix in ['', '_ar_flag', '_gold']:
                if 'belief_facts' in state_key:
                    init_states[state_key+suffix] = {tf: [] for tf in batch[0]['init_state'][state_key]}
                    final_state[state_key+suffix] = {tf: [] for tf in batch[0][self.tgt_state_keys[sk]][state_key]}
                elif 'full_facts' in state_key:
                    init_states[state_key+suffix] = {'true': []}
                    final_state[state_key+suffix] = {'true': []}
            ctxt = []
            for j in range(len(batch)):
                init_state_key = 'init_state'
                init_state, tgt_state = batch[j][init_state_key][state_key], batch[j][self.tgt_state_keys[sk]][state_key]
                if type(init_state) != dict: init_state, tgt_state = {'true': init_state}, {'true': tgt_state}
                if 'fact' in state_key:
                    env = textworld.start(os.path.join(self.gamefile, f'{game_ids[j-i]}.ulx'))
                    game_state = env.reset()
                    inform7_game = env._inform7
                    # game_kb = game_state['game'].kb.inform7_predicates
                    # flagged version for whether state precedes an action or a feedback (in case of `pred_action_and_response_sep`)
                    precedes_action = batch[j]['tgts'].startswith('>')
                    for tf in init_state:
                        init_facts_gold = ' [SEP] '.join(parse_facts_to_nl(init_state[tf], inform7_game))
                        if j >= max_gt_grounded_states: init_facts = ''
                        else: init_facts = init_facts_gold
                        init_states[state_key][tf].append(init_facts)
                        init_states[state_key+'_ar_flag'][tf].append(f"[{'action' if precedes_action else 'feedback'}] "+init_facts)
                        init_states[state_key+'_gold'][tf].append(init_facts_gold)
                    for tf in tgt_state:
                        tgt_facts_gold = ' [SEP] '.join(parse_facts_to_nl(tgt_state[tf], inform7_game))
                        if self.states is not None and tf in self.states[state_key]:
                            # TODO how to find correspondence to batch number????
                            import pdb; pdb.set_trace()
                            tgt_facts = self.states[state_key][tf][batch[j]['data_idx']]
                        elif j >= max_gt_grounded_states:
                            tgt_facts = ''
                        else:
                            tgt_facts = tgt_facts_gold
                        final_state[state_key][tf].append(tgt_facts)
                        final_state[state_key+'_ar_flag'][tf].append(f"[{'action' if precedes_action else 'feedback'}] "+tgt_facts)
                        final_state[state_key+'_gold'][tf].append(tgt_facts_gold)
                elif 'objs' in state_key:
                    can_interact_stmt_init = f"You can interact with {', '.join(init_state['can_interact'])}."
                    can_interact_stmt_final = f"You can interact with {', '.join(tgt_state['can_interact'])}."
                    for suffix in ['', '_ar_flag', '_gold']:
                        if 'can_interact' not in init_states[state_key+suffix]: init_states[state_key+suffix]['can_interact'] = []; final_state[state_key+suffix]['can_interact'] = []
                        init_states[state_key+suffix]['can_interact'].append(can_interact_stmt_init)
                        final_state[state_key+suffix]['can_interact'].append(can_interact_stmt_final)
                # elif 'actions' in state_key:
                #     for tf in init_state: init_states[tf].append(init_state[tf])
                #     for tf in tgt_state: final_state[tf].append(tgt_state[tf])
                else: assert False
                if self.append_facts_to_input:
                    try:
                        ctxt.append(f"{batch[j]['contexts']}{self.tokenizer.convert_ids_to_tokens(self.tokenizer.sep_token_id)}" + final_state['true'][j-i])
                    except: import pdb; pdb.set_trace()
                    
        init_state_tokens = {}
        tgt_state_tokens = {}
        for state_key in init_states:
            init_state_tokens[state_key] = {}
            tgt_state_tokens[state_key] = {}
            for tf in init_states[state_key]:
                tokenized_init_tf = self.tokenizer(init_states[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                tokenized_tgt_tf = self.tokenizer(final_state[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                for k2 in tokenized_init_tf:
                    init_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_init_tf[k2]
                    tgt_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_tgt_tf[k2]
        if self.append_facts_to_input:
            context_tokens = self.tokenizer(ctxt, return_tensors='pt', padding=True, truncation=False).to(DEVICE)
            items_to_keep = context_tokens['attention_mask'].sum(1) <= self.tokenizer.model_max_length
            if not items_to_keep.any():
                return None, None, None, None, game_ids, None
            # Delete problematic example(s) + truncate rest
            context_tokens = {key: apply_mask_and_truncate(context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in context_tokens}
            post_context_tokens = {key: apply_mask_and_truncate(post_context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in post_context_tokens}
            full_context_tokens = {key: apply_mask_and_truncate(full_context_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in full_context_tokens}
            # create target tokens and states, w/out problem examples
            tgt_tokens = self.tokenizer(batch[j]['tgts'], return_tensors='pt', padding=True, truncation=True)
            tgt_tokens = {key: apply_mask_and_truncate(tgt_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in tgt_tokens}
            for state_key in state_keys:
                init_state_tokens[state_key] = {key: apply_mask_and_truncate(init_state_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in init_state_tokens}
                tgt_state_tokens[state_key] = {key: apply_mask_and_truncate(tgt_state_tokens[key], items_to_keep, self.tokenizer.model_max_length) for key in tgt_state_tokens}
        game_ids = [gid for gidx, gid in enumerate(game_ids) if items_to_keep[gidx]]
        context_tokens = {**context_tokens, **{f'post_{key}': post_context_tokens[key] for key in post_context_tokens}, **{f'full_{key}': full_context_tokens[key] for key in full_context_tokens}}
        return context_tokens, tgt_tokens, init_state_tokens, tgt_state_tokens, game_ids, 'all'


def load_possible_pairs(pair_out_file=None, data_dir=None, game_ids=None):
    if pair_out_file and os.path.exists(pair_out_file):
        possible_pairs_serialized = json.load(open(pair_out_file))
        # deserialize
        possible_pairs = {}
        for gameid in possible_pairs_serialized:
            possible_pairs[gameid] = [EntitySet.deserialize(pair_str) for pair_str in possible_pairs_serialized[gameid]]
        return possible_pairs
    elif game_ids is not None:
        return gen_possible_pairs(data_dir, game_ids)[0]
    else:
        return None


def load_negative_tgts(negative_tgts_fn=None, data_dir=None, ent_set_size=None, tokenizer=None, game_ids=None):
    if negative_tgts_fn and os.path.exists(negative_tgts_fn):
        negative_tgts_serialized = torch.load(negative_tgts_fn)
        # negative_tgts = {}
        # for key in negative_tgts_serialized:
        #     negative_tgts[key] = {}
        #     for entset_serialize in negative_tgts_serialized[key]:
        #         negative_tgts[key][EntitySet.deserialize(entset_serialize)] = negative_tgts_serialized[key][entset_serialize]
        return negative_tgts_serialized
    elif game_ids is not None:
        return gen_negative_tgts(data_dir, state_encoder, None, tokenizer, game_ids, ent_set_size)
    else:
        return None


def apply_mask_and_truncate(tensor, mask, max_len,):
    """
    tensor (bsz, seqlen, *)
    mask (bsz)
    max_len (int)
    """
    return tensor[mask][:,:max_len].to(DEVICE)