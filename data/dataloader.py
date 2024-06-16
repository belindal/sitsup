from tqdm import tqdm
import json
import logging
import os
import pdb
import torch
from tqdm import tqdm
import textworld
from textworld.logic import parser
from textworld.logic import Signature, Proposition, Action, Variable, Type
from utils import DEVICE, parse_facts_to_nl, pad_stack
import itertools
import glob
import random


class EntitySets():
    # (unordered) sets of entities
    def __init__(self, entities):
        self.entities = list(entities)

    def __hash__(self):
        return    

    def __eq__(self, other):
        return set(self.entities) == set(other.entities)
    
    def __str__(self):
        return str(self.entities)
    
    def __getitem__(self, i):
        return self.entities[i]


def load_data(dir_path, tokenizer, max_seq_len, max_data_size=10000, inform7_game=None, interleave_state_in_ctxt=False, pred_action_and_response_joint=True, randseed=None, control_input: bool = False):
    # TODO divide by a local window of tokens up to `max_seq_len`
    # for fp in os.listdir(dir_path):
    # [contexts, next utterance]
    full_data = {'contexts': [], 'post_contexts': [], 'tgts': [], 'final_state': [], 'init_state': [], 'filenames': []}  # goal + init state + actions
    actions_data = {'contexts': [], 'post_contexts': [], 'tgts': [], 'final_state': [], 'init_state': [], 'filenames': []}  # actions only
    init_actions_data = {'contexts': [], 'post_contexts': [], 'tgts': [], 'final_state': [], 'init_state': [], 'filenames': []}  # init state + actions
    n_states = 0
    files = glob.glob(os.path.join(dir_path, "*_states.txt"))  # TODO make sorted
    if randseed:
        random.seed(randseed)
        random.shuffle(files)
    for fp in tqdm(files):
        all_actions = []  # actions that make up current file
        curr_action = []  # lines that make up current action
        n_cutoff_actions = 0  # num actions for max_seq_len (not strictly necessary, just ensures we don't run for too long)
        states = []
        # if not os.path.exists(os.path.join(dir_path, f"{fp}.txt")): continue
        # create all_actions (file, separated by commands, aka '>')
        langs_file = fp.replace('_states.txt', '.txt')
        with open(langs_file) as f:
            approx_num_toks = 0
            for line in f:
                if (line.strip().startswith("***") and line.strip().endswith("***")) or approx_num_toks > 2*max_seq_len:
                    # loop will always end on this condition, since "The End" is in all documents
                    break
                line = line.strip() + ' | '
                if line.startswith(">"):
                    action = ''.join(curr_action)
                    if approx_num_toks <= max_seq_len: n_cutoff_actions += 1
                    all_actions.append(action)
                    curr_action = []
                curr_action.append(line)
                approx_num_toks += len(tokenizer.tokenize(line))
                if not pred_action_and_response_joint and line.startswith(">"):
                    # if action, add line immediately
                    action = ''.join(curr_action)
                    if approx_num_toks <= max_seq_len: n_cutoff_actions += 1
                    all_actions.append(action)
                    curr_action = []
            # get last part
            if line.startswith(">") and approx_num_toks + len(tokenizer.tokenize(line)) <= 2*max_seq_len:
                all_actions.append(''.join(curr_action))
                if approx_num_toks + len(tokenizer.tokenize(line)) <= max_seq_len: n_cutoff_actions += 1
        # create final_states
        with open(fp) as f:
            num_lines = 0
            for line in f:
                if num_lines > n_cutoff_actions + 1:  #+1 for initial state
                    break
                state = json.loads(line)
                new_state = {}
                for k in state:
                    if k == 'valid_actions':
                        new_state['valid_actions'] = [vu.replace('\n', ' | ') for vu in state['valid_actions']]
                    elif k == 'invalid_actions':
                        new_state['invalid_actions'] = [vu.replace('\n', ' | ') for vu in state['invalid_actions']]
                    else:
                        new_state[k] = state[k]
                states.append(new_state)
                num_lines += 1

        if interleave_state_in_ctxt:
            all_actions = [
                f"{all_actions[c]}[{'. '.join(parse_facts_to_nl(states[c]['added_belief_facts']['true'], inform7_game))}]\n"
                for c in range(n_cutoff_actions)
            ]
            
        # create (context, next utterance, init_state, states) tuples for each dataset from all_actions
        # (all_actions[0], all_actions[1], states[0], states[0]);
        # (all_actions[0:1], all_actions[2], states[0], states[1]);
        # (all_actions[0:2], all_actions[3], states[0], states[2]);
        # ...
        # NOTE states[i] is state *after* `i`th action, so use (i-1) to get state immediately after context (actions 1...i-1)
        # if 
        interacted_entities = set()
        s = 0  # after all_actions[0]
        for c in range(2,n_cutoff_actions):
            world = os.path.split(langs_file)
            world = os.path.join(os.path.split(world[0])[1], world[1])
            actions = ''.join(all_actions[1:c])
            tgt_action = all_actions[c].split('[')[0]
            postfix = ''.join(all_actions[c:])
            if len(postfix) == 0: import pdb; pdb.set_trace()
            # flag for if state has changed
            increment_corresponding_state = all_actions[c-1].startswith(">")  # last action in context
            if increment_corresponding_state:
                s += 1
                n_states += 1

            full_data['contexts'].append(''.join([all_actions[0], actions]))
            full_data['post_contexts'].append(postfix)
            full_data['tgts'].append(tgt_action)
            full_data['init_state'].append(states[0])
            full_data['final_state'].append(states[s])
            full_data['filenames'].append(world)

            actions_data['contexts'].append(''.join(actions))
            actions_data['post_contexts'].append(postfix)
            actions_data['tgts'].append(tgt_action)
            actions_data['init_state'].append(states[0])
            actions_data['final_state'].append(states[s])
            actions_data['filenames'].append(world)

            goal = all_actions[0].split(' | ')[0]
            if control_input:
                curr_room_fact = None
                curr_room = None
                for curr_room_fact in states[s]['curr_room_belief_facts']['true']:
                    if curr_room_fact['name'] == 'at' and {'name': 'P', 'type': 'P'} in curr_room_fact['arguments']:
                        break
                for entity in curr_room_fact['arguments']:
                    if entity['type'] == 'r':
                        curr_room = entity['name']
                        break
                all_known_rooms = set()
                all_known_objs = set()
                for fact in states[s]['curr_state_belief_facts']['true']:
                    for entity in fact['arguments']:
                        if entity['type'] == 'r':
                            if entity['name'] != curr_room: all_known_rooms.add(entity['name'])
                        elif entity['type'] != 'P':
                            all_known_objs.add(entity['name'])
                all_known_rooms = list(all_known_rooms)
                all_known_objs = list(all_known_objs)
                curr_context = f'You are in the {curr_room}. Other rooms are: '+', '.join(all_known_rooms)+'. Known entities are: '+', '.join(all_known_objs)
                all_actions[0].replace(goal, "")
            else:
                curr_context = ''.join([all_actions[0].replace(goal, ""), actions])
            init_actions_data['contexts'].append(curr_context)
            init_actions_data['post_contexts'].append(postfix)
            init_actions_data['tgts'].append(tgt_action)
            init_actions_data['init_state'].append(states[0])
            init_actions_data['final_state'].append(states[s])
            init_actions_data['filenames'].append(world)

            if len(full_data['contexts']) >= max_data_size:
                break
        if len(full_data['contexts']) >= max_data_size:
            break
        assert s == c // 2
    for k in init_actions_data:
        try: assert len(init_actions_data[k]) == len(init_actions_data['contexts'])
        except: import pdb; pdb.set_trace()
    # print(f"Using files order: {init_actions_data['filenames']}")
    return init_actions_data#, full_data, actions_data


def split_data_by_final_state(data, dev_prop_name, dev_prop_arg_types):
    train_data = {k: [] for k in data}
    dev_data = {k: [] for k in data}
    # split based on state
    for i, final_state in enumerate(data['final_state']):
        has_prop = False
        for p, prop_ser in enumerate(final_state['belief_facts']['true']):
            prop_arg_types = [arg['type'] for arg in prop_ser['arguments']]
            if prop_ser['name'] == dev_prop_name and set(prop_arg_types) == set(dev_prop_arg_types):
                has_prop = True
                break
        if has_prop:
            for k in data: dev_data[k].append(data[k][i])
        else:
            for k in data: train_data[k].append(data[k][i])
    return train_data, dev_data


def get_relevant_facts_about(entities, facts, curr_world=None, entity=None, excluded_entities=None, exact_arg_count=True):
    '''
    entities: list of entities that should *all* appear in list of facts to get (except for the `None` elements)
    excluded_entities: list of entities that should *never* appear in list of facts to get (overrides `entities`)
    exact_arg_count: only get facts with the exact # of non-None arguments as passed-in `entities`
    '''
    relevant_facts = []
    count_nonNone_entities = len([e for e in entities if e is not None])
    for fact in facts:
        exclude_fact = False
        fact_argnames = [arg['name'] for arg in fact['arguments']]
        if exact_arg_count and len(fact_argnames) != count_nonNone_entities: continue  # argument count doesn't match
        if "I" in fact_argnames: fact_argnames[fact_argnames.index("I")] = "inventory"
        if "P" in fact_argnames: fact_argnames[fact_argnames.index("P")] = "player"
        # check none of `excl_entity` shows up in `fact`
        if excluded_entities is not None:
            for excl_entity in excluded_entities:
                # if excl_entity == 'P' or excl_entity == 'I': excl_entity = 'player'
                if excl_entity in fact_argnames:
                    exclude_fact = True
                    break
        if exclude_fact: continue
        add_fact = True
        # if exact_arg_count, entities must appear in (correct position of) fact
        # otherwise, entities must appear (anywhere) in fact
        for e, entity in enumerate(entities):
            if entity is not None and ((exact_arg_count and entity != fact_argnames[e]) or (not exact_arg_count and entity not in fact_argnames)):
                add_fact = False
                continue
        if add_fact: relevant_facts.append(fact)
    return relevant_facts


# def sample_entity_list(all_entities_list, last_cmd, data, tgt, tgt_state_key, state_key, i, n_samples=1):
#     """
#     returns min(n_samples, len(all_entities_list)) samples
#     """
#     def get_entities_in_action(action, preposition, all_entities_list):
#         action = action.split('\n')[0]
#         action = ' '.join(action.split(' ')[2:])
#         if preposition: action = action.split(f' {preposition} ')
#         else: action = [action]
#         assert len(action) == 1 or len(action) == 2
#         obj = action[-1]
#         new_all_entities_list = []
#         # changed obj's relationship to inventory
#         if (obj, 'inventory') in all_entities_list or ('inventory', obj) in all_entities_list: new_all_entities_list.append(('inventory', obj))
#         # changed a property of object (open, eaten)
#         if (None, obj) in all_entities_list: new_all_entities_list.append((None, obj))
#         if len(action) > 1:
#             prep_obj = action[0]
#             if not (prep_obj, obj) in all_entities_list and not (obj, prep_obj) in all_entities_list:
#                 obj = obj.split(' ')[-1]
#             try: assert (prep_obj, obj) in all_entities_list or (obj, prep_obj) in all_entities_list
#             except: import pdb; pdb.set_trace()
#             new_all_entities_list.append((obj, prep_obj))
#         return new_all_entities_list

#     n_samples = min(n_samples, len(all_entities_list))
#     tgt_state = data[tgt_state_key][i][state_key]
#     curr_state_facts = {tf: {json.dumps(fact) for fact in data[tgt_state_key][i][state_key][tf]} for tf in data[tgt_state_key][i][state_key]}

#     get_from_tgt = True
#     """
#     if i == len(data[tgt_state_key]) - 1:
#         all_entities_list = random.sample(all_entities_list, n_samples)
#         if len(all_entities_list[0]) > 2:
#             import pdb; pdb.set_trace()
#         # choose random
#         return all_entities_list, tgt_state
#     # if tgt[0].startswith('> go') and :
#     # import pdb; pdb.set_trace()
#     next_state_facts = {tf: {json.dumps(fact) for fact in data[tgt_state_key][i+1][state_key][tf]} for tf in data[tgt_state_key][i+1][state_key]}
#     changed_facts = next_state_facts['true'].symmetric_difference(curr_state_facts['true']).union(next_state_facts['false'].symmetric_difference(curr_state_facts['false']))
#     # get acording to the next action
#     if len(curr_state_facts['true'].intersection(changed_facts)) == 0: get_from_tgt = True
#     if not get_from_tgt:
#         if tgt.startswith('> go'):
#             # take directional fact
#             # direction = last_cmd[0].split(' ')[2]
#             # location = last_cmd[1][3:-3].lower()
#             direction = tgt.split('\n')[0].split(' ')[2]
#             location = tgt.split('\n')[1][3:-3].lower()
#             found_fact = False
#             for fact in curr_state_facts['true']:
#                 fact = json.loads(fact)
#                 if fact['name'] == f"{direction}_of" and (fact['arguments'][0]['name'] == location or fact['arguments'][1]['name'] == location):
#                     found_fact = True
#                     break
#             if not found_fact:
#                 for fact in curr_state_facts['true']:
#                     fact = json.loads(fact)
#                     if fact['name'] == "at" and fact['arguments'][0]['name'] == 'P': break
#             entity_list = ['player' if arg['name'] == 'P' else arg['name'] for arg in fact['arguments']]
#             entity_list = ['inventory' if arg == 'I' else arg for arg in entity_list]
#             if len(entity_list) == 1: entity_list = [None, entity_list[0]]
#         else:
#             # take first fact
#             for fact in curr_state_facts['true'].intersection(changed_facts):
#                 fact_loaded = json.loads(fact)
#                 entity_list = ['player' if arg['name'] == 'P' else arg['name'] for arg in fact_loaded['arguments']]
#                 entity_list = ['inventory' if arg == 'I' else arg for arg in entity_list]
#                 if len(entity_list) == 1: entity_list = [None, entity_list[0]]
#                 if fact in changed_facts and (entity_list[0], entity_list[1]) in all_entities_list or (entity_list[1], entity_list[0]) in all_entities_list: break
#         if not ((entity_list[0], entity_list[1]) in all_entities_list or (entity_list[1], entity_list[0]) in all_entities_list):
#             get_from_tgt = True
#         else:
#             new_all_entities_list = [(entity_list[0], entity_list[1])]
#     """
#     if get_from_tgt:
#         if tgt.startswith('> go'):
#             tgt_state = data[tgt_state_key][i][state_key]
#             direction = tgt.split('\n')[0].split(' ')[2]
#             location = tgt.split('\n')[1][3:-3].lower()
#             new_all_entities_list = []
#             """
#             # add direction facts to "curr" set of facts in this case
#             next_state = data[tgt_state_key][i+1][state_key]
#             found_fact = False
#             for fact in next_state['true']:
#                 if fact['name'] == f"{direction}_of" and (fact['arguments'][0]['name'] == location or fact['arguments'][1]['name'] == location):
#                     found_fact = True
#                     break
#             if found_fact:
#                 # add the directions to curr state
#                 tgt_state['true'].append(fact)
#                 for fact in next_state['false']:
#                     if fact['name'] in ["east_of", "north_of", "west_of", "south_of"] and (
#                         fact['arguments'][0]['name'] == location or fact['arguments'][1]['name'] == location
#                     ):
#                         tgt_state['false'].append(fact)
#                 if len(fact['arguments']) == 1: entity_list = (None, fact['arguments'][0]['name'])
#                 else: entity_list = (fact['arguments'][0]['name'], fact['arguments'][1]['name'])
#                 new_all_entities_list.append(entity_list)
#             """
#             for fact in tgt_state['true']:
#                 if fact['name'] == "at" and fact['arguments'][0]['name'] == 'P': break
#             entity_list = ['player' if arg['name'] == 'P' else arg['name'] for arg in fact['arguments']]
#             entity_list = ['inventory' if arg == 'I' else arg for arg in entity_list]
#             new_all_entities_list.append((entity_list[0], entity_list[1]))
#         elif tgt.startswith("> open") or tgt.startswith("> close") or tgt.startswith("> unlock") or tgt.startswith("> lock"):
#             new_all_entities_list = get_entities_in_action(tgt, 'with', all_entities_list)
#         elif tgt.startswith("> take"):
#             new_all_entities_list = get_entities_in_action(tgt, 'from', all_entities_list)
#         elif tgt.startswith("> insert"):
#             new_all_entities_list = get_entities_in_action(tgt, 'into', all_entities_list)
#         elif tgt.startswith("> drop") or tgt.startswith("> put"):
#             new_all_entities_list = get_entities_in_action(tgt, 'on', all_entities_list)
#         elif tgt.startswith("> eat"):
#             new_all_entities_list = get_entities_in_action(tgt, None, all_entities_list)
#         elif tgt.startswith("> look"):
#             n_samples = min(n_samples, len(all_entities_list))
#             new_all_entities_list = random.sample(all_entities_list, n_samples)
#         elif tgt.startswith("> inventory") or tgt.startswith("> examine"):
#             if tgt.startswith("> inventory"): obj = "inventory"
#             elif tgt.startswith("> examine"): obj = " ".join(tgt.split(" ")[2:]).split('\n')[0]
#             # get all entities/pairs that include obj
#             new_all_entities_list = [entity_list for entity_list in all_entities_list if obj in entity_list]

#         """
#         # check all entities mentioned in `tgt`
#         # next_cmd = tgt.split('\n')[0]
#         longest_mentioned_entity_lists = None
#         for entity_list in all_entities_list:
#             all_in_list = True
#             for entity in entity_list:
#                 if entity == None: continue
#                 else: all_in_list &= entity in tgt
#             if all_in_list:
#                 pruned_entity_list = [ent for ent in entity_list if ent is not None]
#                 if longest_mentioned_entity_lists is None or len(pruned_entity_list) > len(longest_mentioned_entity_lists):
#                     longest_mentioned_entity_lists = pruned_entity_list
#                 elif len(pruned_entity_list) == len(longest_mentioned_entity_lists) and len(''.join(pruned_entity_list)) > len(''.join(longest_mentioned_entity_lists)):
#                     longest_mentioned_entity_lists = pruned_entity_list
#         if longest_mentioned_entity_lists is None: import pdb; pdb.set_trace()
#         if len(longest_mentioned_entity_lists) == 1: longest_mentioned_entity_lists = [None, longest_mentioned_entity_lists[0]]
#         new_all_entities_list = [longest_mentioned_entity_lists]
#         """
#     return new_all_entities_list, tgt_state


def convert_to_transformer_batches(
    args, data, tokenizer, batchsize, train_state=None,
    inform7_game=None, control_input=False, possible_pairs=None,
    training=False, game_id_to_entities=None, append_facts_to_input=False,
    # TODO make this parameter something better
    nnegs=0, npos=1, include_feedback=False, expected_states=None,
    max_gt_grounded_states=float("inf"),
):
    """
    train_state: [state_add_type, key]
    nnegs: # negatives to get (set 0 to not get negatives, inf to get all negatives)
    npos: # positives to get (default 1)
    expected_states: to use for EM (in cases where gold-annotated states are unavailable)
    """
    # TODO just 1 key to  make this code less messy...
    def apply_mask_and_truncate(tensor, mask, max_len,):
        """
        tensor (bsz, seqlen, *)
        mask (bsz)
        max_len (int)
        """
        return tensor[mask][:,:max_len].to(DEVICE)

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    if train_state:
        train_state[1] = train_state[1].replace('_single', '').replace('_pair', '')

    batch_num = 0
    for i in range(0, len(data['contexts']), batchsize):
        game_ids = [fn.split('_')[0] for fn in data['filenames'][i:i+batchsize]]
        context_tokens = tokenizer(data['contexts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=False)
        post_context_tokens = tokenizer(data['post_contexts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True, max_length=128)
        full_context_tokens = tokenizer([
            data['contexts'][j] + ' [SEP] ' + data['post_contexts'][j] for j in range(i, min(i+batchsize, len(data['contexts'])))
        ], return_tensors='pt', padding=True, truncation=False)
        items_to_keep = context_tokens['attention_mask'].sum(1) <= tokenizer.model_max_length
        if not items_to_keep.any():
            yield None, None, None, None, game_ids, None
            continue

        # Delete problematic example(s) + truncate rest
        context_tokens = {key: apply_mask_and_truncate(context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in context_tokens}
        post_context_tokens = {key: apply_mask_and_truncate(post_context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in post_context_tokens}
        full_context_tokens = {key: apply_mask_and_truncate(full_context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in full_context_tokens}
        if nnegs > 0 or npos > 1:
            tgt_tokens = tokenizer(data['tgts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True)
            tgt_tokens['all_cands_input_ids'] = []
            tgt_tokens['all_cands_attention_mask'] = []
            tgt_tokens['labels'] = []
            tgt_tokens['valid_actions'] = []
            tgt_tokens['invalid_actions'] = []
            for j in range(i, min(len(data['tgts']), i+batchsize)):
                # do some basic filtering
                valid_inputs = [action for action in data['final_state'][j]['valid_actions'] if not action.startswith('examine') and not action.startswith('look') and not action.startswith('inventory')]
                if len(valid_inputs) == 0:
                    valid_inputs = data['final_state'][j]['valid_actions']
                tgt_tokens['valid_actions'].append(valid_inputs)
                tgt_tokens['invalid_actions'].append(data['final_state'][j]['invalid_actions'])
                # sample
                if training:
                    nnegs_to_sample = min(nnegs, len(data['final_state'][j]['invalid_actions']))
                    npos_to_sample = min(npos, len(valid_inputs))
                else:
                    nnegs_to_sample = min(nnegs, len(data['final_state'][j]['invalid_actions']))
                    npos_to_sample = min(npos, len(valid_inputs))
                if data['tgts'][j].startswith('>'):
                    # is action
                    if include_feedback: valid_inputs = ["> "+vi+' | ' for vi in valid_inputs]
                    else: valid_inputs = ["> "+vi.split(' | ')[0]+' | ' for vi in valid_inputs]
                    if npos_to_sample == 1:
                        valid_inputs = [data['tgts'][j]]
                    else:
                        if data['tgts'][j] in valid_inputs:
                            valid_inputs.remove(data['tgts'][j])
                        valid_inputs = [data['tgts'][j]] + random.sample(valid_inputs, npos_to_sample-1)
                    invalid_inputs = random.sample(data['final_state'][j]['invalid_actions'], nnegs_to_sample)
                    if include_feedback: invalid_inputs = ["> "+ii+' | ' for ii in invalid_inputs]
                    else: invalid_inputs = ["> "+ii.split(' | ')[0]+' | ' for ii in invalid_inputs]
                else:
                    # is feedback -- all other feedbacks are invalid
                    # assert npos_to_sample == 1
                    # if '-=' in data['tgts'][j]: import pdb; pdb.set_trace()
                    valid_inputs = [data['tgts'][j].split(' | ')[0]+' | ']  # current feedback
                    invalid_inputs = [ii.split(' | ')[0]+' | ' for ii in data['tgts'][i:j] + data['tgts'][j+1:i+batchsize] if not ii.startswith('> ')]  # other feedbacks in batch
                    if nnegs_to_sample < len(invalid_inputs):
                        invalid_inputs = random.sample(invalid_inputs, nnegs_to_sample)
                cand_tokens = tokenizer(valid_inputs + invalid_inputs, return_tensors='pt', padding=True, truncation=True)
                for k in cand_tokens: tgt_tokens[f'all_cands_{k}'].append(cand_tokens[k])
                tgt_tokens['labels'].append([1 for _ in valid_inputs] + [0 for _ in invalid_inputs])
            # (bs, n_cands, seqlen)
            tgt_tokens['all_cands_input_ids'], tgt_tokens['all_cands_attention_mask'] = pad_stack(
                tgt_tokens['all_cands_input_ids'], tgt_tokens['all_cands_attention_mask'], pad_idx=tokenizer.pad_token_id, device=DEVICE)
            n_cands = tgt_tokens['all_cands_input_ids'].size(1)
            # (bs, n_cands,)
            tgt_tokens['labels'] = torch.tensor([label + [0 for _ in range(n_cands-len(label))] for label in tgt_tokens['labels']]).to(DEVICE)
        else:
            # create target tokens and states
            tgt_tokens = tokenizer(data['tgts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True)
        # delete problem examples
        tgt_tokens = {key: apply_mask_and_truncate(tgt_tokens[key], items_to_keep, tokenizer.model_max_length) if type(tgt_tokens[key]) == torch.Tensor else tgt_tokens[key] for key in tgt_tokens}
        
        init_states = {}
        final_state = {}
        if train_state:
            for suffix in ['', '_expected', '_ar_flag', '_input', '_gold', '_concat_text']:
                if 'belief_facts' in train_state[1]:
                    init_states[train_state[1]+suffix] = {tf: [] for tf in data['init_state'][0][train_state[1]]}
                    final_state[train_state[1]+suffix] = {tf: [] for tf in data['final_state'][0][train_state[1]]}
                elif 'full_facts' in train_state[1]:
                    init_states[train_state[1]+suffix] = {'true': []}
                    final_state[train_state[1]+suffix] = {'true': []}
            ctxt = []
            for j in range(i,min(i+batchsize, len(data['contexts']))):
                init_state, tgt_state = data['init_state'][j][train_state[1]], data['final_state'][j][train_state[1]]
                if type(init_state) != dict: init_state, tgt_state = {'true': init_state}, {'true': tgt_state}
                if 'fact' in train_state[1]:
                    env = textworld.start(os.path.join(args.gamefile, f'{game_ids[j-i]}.ulx'))
                    game_state = env.reset()
                    inform7_game = env._inform7
                    # game_kb = game_state['game'].kb.inform7_predicates
                    # flagged version for whether state precedes an action or a feedback (in case of `pred_action_and_response_sep`)
                    precedes_action = data['tgts'][j].startswith('>')
                    for tf in init_state:
                        init_facts_gold = ' [SEP] '.join(parse_facts_to_nl(init_state[tf], inform7_game))
                        if j >= max_gt_grounded_states: init_facts = ''
                        else: init_facts = init_facts_gold
                        init_states[train_state[1]][tf].append(init_facts)
                        init_states[train_state[1]+'_expected'][tf].append(init_facts)
                        init_states[train_state[1]+'_ar_flag'][tf].append(f"[{'action' if precedes_action else 'feedback'}] "+init_facts)
                        init_states[train_state[1]+'_input'][tf].append(init_facts+(f" | > {data['contexts'][j].split('>')[-1].strip(' |')}" if not precedes_action else ''))
                        init_states[train_state[1]+'_concat_text'][tf].append(f"{data['contexts'][j]} [SEP] {init_facts}")
                        init_states[train_state[1]+'_gold'][tf].append(init_facts_gold)
                    for tf in tgt_state:
                        tgt_facts_gold = ' [SEP] '.join(parse_facts_to_nl(tgt_state[tf], inform7_game))
                        if j >= max_gt_grounded_states:
                            tgt_facts = ''
                            if expected_states is not None and tf in expected_states:
                                expected_facts = expected_states[tf][j]
                            else:
                                expected_facts = ''
                        else:
                            tgt_facts = tgt_facts_gold
                            expected_facts = tgt_facts_gold
                        final_state[train_state[1]][tf].append(tgt_facts)
                        final_state[train_state[1]+'_expected'][tf].append(expected_facts)
                        final_state[train_state[1]+'_ar_flag'][tf].append(f"[{'action' if precedes_action else 'feedback'}] "+tgt_facts)
                        final_state[train_state[1]+'_input'][tf].append(tgt_facts+(f" | > {data['contexts'][j].split('>')[-1].strip(' |')}" if not precedes_action else ''))
                        final_state[train_state[1]+'_concat_text'][tf].append(f"{data['contexts'][j]} [SEP] {tgt_facts}")
                        final_state[train_state[1]+'_gold'][tf].append(tgt_facts_gold)
                elif 'objs' in train_state[1]:
                    can_interact_stmt_init = f"You can interact with {', '.join(init_state['can_interact'])}."
                    can_interact_stmt_final = f"You can interact with {', '.join(tgt_state['can_interact'])}."
                    for suffix in ['','_expected',  '_ar_flag', '_input' '_concat_text', '_gold']:
                        if 'can_interact' not in init_states[train_state[1]+suffix]: init_states[train_state[1]+suffix]['can_interact'] = []; final_state[train_state[1]+suffix]['can_interact'] = []
                        init_states[train_state[1]+suffix]['can_interact'].append(can_interact_stmt_init)
                        final_state[train_state[1]+suffix]['can_interact'].append(can_interact_stmt_final)
                # elif 'actions' in train_state[1]:
                #     for tf in init_state: init_states[tf].append(init_state[tf])
                #     for tf in tgt_state: final_state[tf].append(tgt_state[tf])
                else: assert False
                if append_facts_to_input:
                    try:
                        ctxt.append(f"{data['contexts'][j]}{tokenizer.convert_ids_to_tokens(tokenizer.sep_token_id)}" + final_state['true'][j-i])
                    except: import pdb; pdb.set_trace()

        init_state_tokens = {}
        tgt_state_tokens = {}
        for state_key in init_states:
            init_state_tokens[state_key] = {}
            tgt_state_tokens[state_key] = {}
            for tf in init_states[state_key]:
                # if 'actions' in state_key:
                #     # (bs, n_cands, seqlen)
                #     tgt_tokens['all_cands_input_ids'], tgt_tokens['all_cands_attention_mask'] = pad_stack(
                #         tgt_tokens['all_cands_input_ids'], tgt_tokens['all_cands_attention_mask'], pad_idx=tokenizer.pad_token_id, device=DEVICE)
                #     n_cands = tgt_tokens['all_cands_input_ids'].size(1)
                tokenized_init_tf = tokenizer(init_states[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                tokenized_tgt_tf = tokenizer(final_state[state_key][tf], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
                for k2 in tokenized_init_tf:
                    init_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_init_tf[k2]
                    tgt_state_tokens[state_key][f'{k2}{"_"+tf if tf != "true" else ""}'] = tokenized_tgt_tf[k2]
                # if get_negatives:
                #     tokenized_init = {k: tokenized_init[k].unsqueeze(1).expand(
                #         -1, nnegs+npos, *[-1 for _ in range(len(tokenized_init[k].size())-1)],
                #     ) for k in tokenized_init}
                #     tokenized_tgt = {k: tokenized_tgt[k].unsqueeze(1).expand(
                #         -1, nnegs+npos, *[-1 for _ in range(len(tokenized_tgt[k].size())-1)],
                #     ) for k in tokenized_tgt}
            # init_state_tokens[state_key] = {**init_state_tokens[state_key], **tokenized_init}
            # tgt_state_tokens[state_key] = {**tgt_state_tokens[state_key], **tokenized_tgt}
        if append_facts_to_input:
            context_tokens = tokenizer(ctxt, return_tensors='pt', padding=True, truncation=False).to(DEVICE)
            # TODO truncation???
            items_to_keep = context_tokens['attention_mask'].sum(1) <= tokenizer.model_max_length
            if not items_to_keep.any():
                yield None, None, None, None, game_ids, None
                continue
            # Delete problematic example(s) + truncate rest
            context_tokens = {key: apply_mask_and_truncate(context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in context_tokens}
            post_context_tokens = {key: apply_mask_and_truncate(post_context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in post_context_tokens}
            full_context_tokens = {key: apply_mask_and_truncate(full_context_tokens[key], items_to_keep, tokenizer.model_max_length) for key in full_context_tokens}
            # create target tokens and states, w/out problem examples
            tgt_tokens = tokenizer(data['tgts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True)
            tgt_tokens = {key: apply_mask_and_truncate(tgt_tokens[key], items_to_keep, tokenizer.model_max_length) for key in tgt_tokens}
            if train_state[1]:
                init_state_tokens[train_state[1]] = {key: apply_mask_and_truncate(init_state_tokens[key], items_to_keep, tokenizer.model_max_length) for key in init_state_tokens}
                tgt_state_tokens[train_state[1]] = {key: apply_mask_and_truncate(tgt_state_tokens[key], items_to_keep, tokenizer.model_max_length) for key in tgt_state_tokens}
        game_ids = [gid for gidx, gid in enumerate(game_ids) if items_to_keep[gidx]]
        context_tokens = {**context_tokens, **{f'post_{key}': post_context_tokens[key] for key in post_context_tokens}, **{f'full_{key}': full_context_tokens[key] for key in full_context_tokens}}
        yield context_tokens, tgt_tokens, init_state_tokens, tgt_state_tokens, game_ids, 'all'


# def get_control_pairs():
#     '''
#     Generate entity pairing for control experiments
#     (entity to ask about, entity mention)
#     '''
#     # TODO not hard-coded
ENTITIES_SIMPLE = {'player', 'inventory', 'wooden door', 'chest drawer', 'antique trunk', 'king-size bed', 'old key', 'lettuce', 'tomato plant', 'milk', 'shovel', 'toilet', 'bath', 'sink', 'soap bar', 'toothbrush', 'screen door', 'set of chairs', 'bbq', 'patio table', 'couch', 'low table', 'tv', 'half of a bag of chips', 'remote', 'refrigerator', 'counter', 'stove', 'kitchen island', 'bell pepper', 'apple', 'note'}
ENTITIES_TH = {
    'box', 'chest', 'stand', 'chair', 'bookshelf', 'fondue', 'passkey', 'refrigerator', 'saucepan', 'drawer', 'dresser', 'cabinet', 'case', 'shelf', 'bench', 'table', 'fridge', 'bowl', 'pan', 'trunk', 'bureau', 'mantelpiece', 'gummy bear', 'safe', 'couch', 'coffer', 'locker', 'suitcase', 'bed', 'poem', 'board', 'keycard', 'portmanteau', 'basket', 'mantle', 'bed stand', 'desk', 'latchkey', 'gate', 'insect', 'hatch', 'workbench', 'rusty table', 'bug', 'counter', 'rack', 'platter', 'plate',
    'American style portal', 'American style key', 'display', 'folder', 'crate', 'toolbox', 'shirt', 'type H chest', 'type H keycard', 'nest of caterpillars', 'key', 'portal', 'rectangular portal', 'rectangular key', 'licorice strip', 'bar', 'spherical gateway', 'spherical latchkey', 'recliner', 'chocolate bar', 'grape', 'gateway', 'freezer', 'worm', 'big freezer', 'passageway', 'sandwich', 'mouse', 'armchair', 'formless passkey', 'splintery workbench', 'formless locker', 'apple',
    'textbook', 'peanut', 'formless door', 'stick of butter', 'legume', 'shoe', 'shoddy table', 'lavender scented safe', 'cabbage', 'lavender scented latchkey', 'American limited edition hatch', 'potato', 'American limited edition passkey', 'non-euclidean gate', 'non-euclidean keycard', 'cookie', 'spherical hatch', 'door', 'nest of bugs', 'spherical keycard', 'type Z door', 'type Z latchkey', 'nest of insects', 'cuboid locker', 'cuboid key', 'American style hatch', 'cucumber', 'paper towel',
    'non-euclidean gateway', 'rough stand', 'Quote of the Day Calendar', 'Canadian style passageway', 'Canadian style latchkey', 'cauliflower', 'formless passageway', 'American locker', 'American style chest', 'American style passkey', 'loaf of bread', 'formless box', 'formless latchkey', 'cake scented portal', 'dusty shelf', 'cake scented passkey', 'Microsoft gateway', 'Microsoft keycard', 'rusty counter', 'durian', 'broccoli', 'Canadian hatch', 'formless keycard', 'coconut', 'formless safe',
    'Canadian latchkey', 'spherical gate', 'splintery counter', 'shoddy counter', 'berry', 'cashew', 'non-euclidean key', 'non-euclidean safe', 'lightbulb', 'teapot', 'TextWorld style key', 'TextWorld style door', 'butterfly', 'type A hatch', 'type A latchkey', 'cd', 'formless chest', 'shiny counter', 'Microsoft style chest', 'Microsoft style keycard', "Henderson's style portal", "Henderson's style keycard", 'rough table', 'American box', 'book', 'American latchkey', 'type R safe', 'type R passkey',
    "Henderson's style key", "Henderson's style safe", 'gojiberry', 'knife', 'American gateway', 'American keycard', 'top hat', 'type O gate', 'keyboard', 'splintery stand', 'type O latchkey', 'Microsoft limited edition hatch', 'Microsoft limited edition latchkey', 'dusty rack', 'spherical passkey', 'spherical chest', 'chipped board', 'fresh laundry scented passageway', 'fresh laundry scented key', 'gross mantelpiece', 'fancy cabinet', 'big fridge', 'big refrigerator', 'teaspoon', 'nest of kittens',
    'formless gate', 'nest of grubs', 'Microsoft limited edition key', 'candy bar', 'dusty table', 'disk', 'salad', 'broom', 'type I passageway', 'splintery table', 'type I key', 'rectangular locker', 'rectangular passkey', 'Cat Calendar', 'Microsoft portal', 'spherical locker', 'shiny rack', 'shiny bench', 'telephone', 'shiny table', 'fork', 'type 3 passageway', 'fly larva', 'type 3 passkey', 'fancy case', 'small refrigerator', 'small freezer', 'non-euclidean latchkey', 'rectangular latchkey',
    'rough shelf', 'shoddy rack', 'spherical portal', 'spherical key', 'rectangular hatch', 'American style latchkey', 'nest of spiders', 'burger', 'type 8 safe', 'type 8 passkey', 'type Y door', 'soap dispenser', 'type Y passkey', 'non-euclidean hatch', 'small chest', 'shiny board', 'monitor', 'pizza', 'TextWorld style gate', 'type P passageway', 'shadfly', 'type P latchkey', 'laptop', "Henderson's limited edition gate", "Henderson's limited edition keycard", 'type 3 box', 'rectangular keycard',
    'type 3 latchkey', 'rectangular gateway', 'spork', 'soap scented passageway', 'soap scented latchkey', 'lavender scented passageway', 'lavender scented passkey', 'sponge', 'Canadian portal', 'Canadian key', 'shoddy shelf', 'American limited edition door', 'American limited edition key', 'fudge scented gateway', 'fudge scented key', 'neglected locker', 'chipped bench', 'chipped table', 'mop', 'frisbee', 'American gate', 'dvd', 'type T gateway', 'type T passkey', 'melon', 'Canadian gateway', 'plant',
    'rough rack', 'formless key', 'rusty stand', 'shoddy stand', 'shoddy workbench', 'type L passageway', 'rough counter', 'rectangular passageway', 'formless portal', 'cane', 'Canadian box', 'pillow', 'splintery rack', 'Microsoft style portal', 'Microsoft style passkey', 'iron', 'Microsoft limited edition chest', 'type 5 gateway', 'teacup', 'type 5 key', 'cuboid gate', 'cuboid passkey', 'non-euclidean chest', 'cloak', 'cuboid latchkey', 'cuboid hatch', 'cuboid portal', 'small case', 'type 6 latchkey',
    'type 6 box', 'type W box', 'type W passkey', 'Canadian limited edition portal', 'Canadian limited edition passkey', 'type C safe', 'type C key', 'TextWorld style chest', 'non-euclidean door', 'TextWorld limited edition gateway', 'TextWorld limited edition keycard', 'glass', 'garlic clove', 'gooseberry', 'chipped counter', 'dusty board', 'type 1 chest', 'type 1 passkey', 'spherical passageway', 'Canadian style gate', 'Canadian style key', 'Canadian chest', 'scarf', 'type Q passageway', 'rusty workbench',
    'rusty rack', 'rough workbench', 'Microsoft style gateway', 'pair of pants', 'Microsoft style latchkey', 'American limited edition safe', 'American limited edition latchkey', 'dusty counter', 'new chest', 'new basket', 'tablet', 'spherical safe', 'cake scented safe', 'cake scented key', 'blender', 'cuboid box', 'Canadian limited edition hatch', 'fudge scented passkey', 'fudge scented portal', 'fancy refrigerator', 'rectangular gate', "Henderson's style chest", 'sock', 'type H gate', 'type Z portal',
    'type 2 passageway', 'type 2 latchkey', 'type Z keycard', 'fancy freezer', 'spoon', 'pear', 'mug', 'birchwood drawer', 'hat', 'Microsoft style door', 'onion', 'fancy chest', 'glove', 'rectangular chest', 'blanket', 'banana', 'type O portal', 'splintery shelf', 'cake scented passageway', 'TextWorld style portal', 'type 8 passageway', 'type 8 key', 'rectangular door', 'cuboid door', 'manuscript', 'non-euclidean portal', 'type 4 gateway', 'type 4 passkey', 'mat', 'fresh laundry scented portal',
    'vanilla scented keycard', 'vanilla scented passageway', 'fresh laundry scented passkey', 'type Y chest', 'printer', 'type Y key', 'shiny shelf', 'vanilla scented gateway', 'vanilla scented key', 'Comic Strip Calendar', 'worn table', 'Canadian limited edition safe', 'Canadian limited edition key', 'lavender scented door', 'lavender scented keycard', 'cuboid keycard', 'nest of bunnies', 'walnutwood cabinet', "Henderson's hatch", "Henderson's passkey", 'Canadian limited edition gate', 'spherical box', 'Canadian limited edition latchkey',
}
ROOMS_SIMPLE = {'garden', 'bathroom', 'kitchen', 'bedroom', 'backyard', 'living room'}
ROOMS_TH = {'study', 'bedroom', 'office', 'recreation zone', 'shower', 'vault', 'kitchen', 'cookery', 'laundry place', 'sauna', 'dish-pit', 'bedchamber', 'parlor', 'spare room', 'launderette', 'basement', 'chamber', 'canteen', 'attic', 'scullery', 'cellar', 'cubicle', 'cookhouse', 'kitchenette', 'lounge', 'pantry', 'garage', 'closet', 'bar', 'salon', 'studio', 'workshop', 'steam room', 'bathroom', 'washroom', 'laundromat', 'restroom', 'playroom'}

# TODO make this more official
ENTITIES = ENTITIES_SIMPLE.union(ENTITIES_TH).union({'player', 'inventory'})
ROOMS = ROOMS_SIMPLE.union(ROOMS_TH).union({'player', 'inventory'})
control_pairs_simple = [
    ('player', 'inventory'), ('inventory', 'player'), ('wooden door', 'screen door'), ('screen door', 'refrigerator'), ('refrigerator', 'counter'), ('counter', 'stove'),
    ('stove', 'kitchen island'), ('kitchen island', 'apple'), ('apple', 'note'), ('note', 'tomato plant'), ('tomato plant', 'wooden door'), ('bell pepper', 'milk'), ('milk', 'shovel'),
    ('shovel', 'half of a bag of chips'), ('half of a bag of chips', 'bell pepper'), ('toilet', 'bath'), ('bath', 'sink'), ('sink', 'soap bar'), ('soap bar', 'toothbrush'),
    ('toothbrush', 'toilet'), ('lettuce', 'couch'), ('couch', 'low table'), ('low table', 'tv'), ('tv', 'remote'), ('remote', 'lettuce'), ('chest drawer', 'antique trunk'),
    ('antique trunk', 'king-size bed'), ('king-size bed', 'old key'), ('old key', 'chest drawer'), ('set of chairs', 'bbq'), ('bbq', 'patio table'), ('patio table', 'set of chairs')
]
control_pairs_with_rooms_simple = control_pairs_simple + [('garden', 'bathroom'), ('bathroom', 'kitchen'), ('kitchen', 'bedroom'), ('bedroom', 'backyard'), ('backyard', 'living room'), ('living room', 'garden')]
control_pairs_TH = [
    ('box', 'stand'), ('stand', 'mantelpiece'), ('mantelpiece', 'coffer'), ('coffer', 'spherical latchkey'), ('spherical latchkey', 'grape'), ('grape', 'armchair'), ('armchair', 'legume'), ('legume', 'cauliflower'), ('cauliflower', 'formless passageway'), ('formless passageway', 'TextWorld style door'), ('TextWorld style door', 'cd'), ('cd', 'disk'), ('disk', 'TextWorld style gate'), ('TextWorld style gate', 'formless key'), ('formless key', 'Microsoft style gateway'), ('Microsoft style gateway', 'type Y chest'), ('type Y chest', 'printer'),
    ('printer', 'type Y key'), ('type Y key', 'box'), ('chest', 'trunk'), ('trunk', 'couch'), ('couch', 'recliner'), ('recliner', 'non-euclidean gateway'), ('non-euclidean gateway', 'Canadian style passageway'), ('Canadian style passageway', 'formless safe'), ('formless safe', 'American box'), ('American box', 'Canadian box'), ('Canadian box', 'pillow'), ('pillow', 'new chest'), ('new chest', 'new basket'), ('new basket', 'chest'), ('chair', 'case'), ('case', 'bureau'), ('bureau', 'locker'), ('locker', 'portmanteau'), ('portmanteau', 'desk'),
    ('desk', 'display'), ('display', 'folder'), ('folder', 'portal'), ('portal', 'mouse'), ('mouse', 'cuboid locker'), ('cuboid locker', 'American style chest'), ('American style chest', 'American latchkey'), ('American latchkey', 'type I passageway'), ('type I passageway', 'rectangular locker'), ('rectangular locker', 'Cat Calendar'), ('Cat Calendar', 'type 3 latchkey'), ('type 3 latchkey', 'rectangular passageway'), ('rectangular passageway', 'cuboid passkey'), ('cuboid passkey', 'cuboid portal'), ('cuboid portal', 'type 1 chest'),
    ('type 1 chest', 'type 1 passkey'), ('type 1 passkey', 'Canadian style gate'), ('Canadian style gate', 'cuboid box'), ('cuboid box', 'type 4 passkey'), ('type 4 passkey', 'nest of bunnies'), ('nest of bunnies', 'chair'), ('bookshelf', 'suitcase'), ('suitcase', 'mantle'), ('mantle', 'sandwich'), ('sandwich', 'door'), ('door', 'nest of insects'), ('nest of insects', 'type R safe'), ('type R safe', 'type R passkey'), ('type R passkey', 'spherical passageway'), ('spherical passageway', 'type H gate'), ('type H gate', 'manuscript'),
    ('manuscript', 'vanilla scented key'), ('vanilla scented key', 'bookshelf'), ('fondue', 'dresser'), ('dresser', 'shelf'), ('shelf', 'gummy bear'), ('gummy bear', 'board'), ('board', 'hatch'), ('hatch', 'counter'), ('counter', 'spherical gateway'), ('spherical gateway', 'apple'), ('apple', 'Canadian style latchkey'), ('Canadian style latchkey', 'TextWorld style key'), ('TextWorld style key', 'dusty table'), ('dusty table', 'rectangular passkey'), ('rectangular passkey', 'shiny table'), ('shiny table', 'sponge'), ('sponge', 'Canadian portal'),
    ('Canadian portal', 'chipped bench'), ('chipped bench', 'mop'), ('mop', 'dusty board'), ('dusty board', 'dusty counter'), ('dusty counter', 'shiny shelf'), ('shiny shelf', 'fondue'), ('passkey', 'crate'), ('crate', 'shirt'), ('shirt', 'key'), ('key', 'passageway'), ('passageway', 'splintery stand'), ('splintery stand', 'splintery table'), ('splintery table', 'type P latchkey'), ('type P latchkey', 'American limited edition door'), ('American limited edition door', 'American gate'), ('American gate', 'rough rack'), ('rough rack', 'cloak'),
    ('cloak', 'TextWorld style chest'), ('TextWorld style chest', 'scarf'), ('scarf', 'splintery shelf'), ('splintery shelf', 'type 8 key'), ('type 8 key', 'vanilla scented keycard'), ('vanilla scented keycard', 'passkey'), ('refrigerator', 'platter'), ('platter', 'American gateway'), ('American gateway', 'American keycard'), ('American keycard', 'type 2 passageway'), ('type 2 passageway', 'type 2 latchkey'), ('type 2 latchkey', 'fancy chest'), ('fancy chest', 'refrigerator'), ('saucepan', 'licorice strip'), ('licorice strip', 'formless box'),
    ('formless box', 'knife'), ('knife', 'fancy cabinet'), ('fancy cabinet', 'teaspoon'), ('teaspoon', 'small refrigerator'), ('small refrigerator', 'small freezer'), ('small freezer', 'nest of spiders'), ('nest of spiders', 'burger'), ('burger', 'small case'), ('small case', 'fudge scented passkey'), ('fudge scented passkey', 'saucepan'), ('drawer', 'Canadian latchkey'), ('Canadian latchkey', 'non-euclidean key'), ('non-euclidean key', 'Canadian chest'), ('Canadian chest', 'pear'), ('pear', 'banana'), ('banana', 'spherical box'), ('spherical box', 'drawer'),
    ('cabinet', 'bench'), ('bench', 'table'), ('table', 'safe'), ('safe', 'rack'), ('rack', 'spherical keycard'), ('spherical keycard', 'paper towel'), ('paper towel', 'spherical gate'), ('spherical gate', 'type 3 passkey'), ('type 3 passkey', 'spherical key'), ('spherical key', 'type Y door'), ('type Y door', 'soap dispenser'), ('soap dispenser', 'lavender scented passageway'), ('lavender scented passageway', 'lavender scented passkey'), ('lavender scented passkey', 'chipped table'), ('chipped table', 'type Z portal'), ('type Z portal', 'cabinet'),
    ('fridge', 'bowl'), ('bowl', 'pan'), ('pan', 'plate'), ('plate', 'chocolate bar'), ('chocolate bar', 'freezer'), ('freezer', 'formless locker'), ('formless locker', 'potato'), ('potato', 'cake scented portal'), ('cake scented portal', 'cake scented passkey'), ('cake scented passkey', 'rectangular latchkey'), ('rectangular latchkey', 'soap scented passageway'), ('soap scented passageway', 'type L passageway'), ('type L passageway', 'type Z keycard'), ('type Z keycard', 'spoon'), ('spoon', 'Canadian limited edition gate'),
    ('Canadian limited edition gate', 'Canadian limited edition latchkey'), ('Canadian limited edition latchkey', 'fridge'), ('bed', 'worm'), ('worm', 'American style hatch'), ('American style hatch', 'Microsoft style keycard'), ('Microsoft style keycard', 'Microsoft style portal'), ('Microsoft style portal', 'bed'), ('poem', 'bed stand'), ('bed stand', 'formless passkey'), ('formless passkey', 'rectangular hatch'), ('rectangular hatch', 'Canadian key'), ('Canadian key', 'Canadian gateway'), ('Canadian gateway', 'plant'), ('plant', 'type C safe'),
    ('type C safe', 'type C key'), ('type C key', 'Canadian limited edition hatch'), ('Canadian limited edition hatch', 'poem'), ('keycard', 'peanut'), ('peanut', 'lavender scented latchkey'), ('lavender scented latchkey', 'cuboid key'), ('cuboid key', 'rough stand'), ('rough stand', 'formless latchkey'), ('formless latchkey', 'type I key'), ('type I key', 'vanilla scented gateway'), ('vanilla scented gateway', 'keycard'), ('basket', 'Microsoft style chest'), ('Microsoft style chest', 'chipped board'), ('chipped board', 'salad'),
    ('salad', 'broom'), ('broom', 'type 5 gateway'), ('type 5 gateway', 'type 5 key'), ('type 5 key', 'mat'), ('mat', 'walnutwood cabinet'), ('walnutwood cabinet', 'basket'), ('latchkey', 'workbench'), ('workbench', 'shoe'), ('shoe', 'nest of bugs'), ('nest of bugs', 'Microsoft gateway'), ('Microsoft gateway', 'Canadian hatch'), ('Canadian hatch', 'type O gate'), ('type O gate', 'shoddy rack'), ('shoddy rack', 'rectangular keycard'), ('rectangular keycard', 'cuboid gate'), ('cuboid gate', 'fudge scented portal'), ('fudge scented portal', 'cuboid door'),
    ('cuboid door', 'fresh laundry scented portal'), ('fresh laundry scented portal', 'cuboid keycard'), ('cuboid keycard', 'latchkey'), ('gate', 'rectangular key'), ('rectangular key', 'textbook'), ('textbook', 'lavender scented safe'), ('lavender scented safe', 'spherical hatch'), ('spherical hatch', 'Microsoft keycard'), ('Microsoft keycard', 'nest of kittens'), ('nest of kittens', 'Microsoft portal'), ('Microsoft portal', 'monitor'), ('monitor', 'type P passageway'), ('type P passageway', 'Microsoft style door'), ('Microsoft style door', 'type 4 gateway'),
    ('type 4 gateway', 'gate'), ('insect', 'American style key'), ('American style key', 'candy bar'), ('candy bar', 'fork'), ('fork', 'type 8 safe'), ('type 8 safe', 'type 8 passkey'), ('type 8 passkey', 'pizza'), ('pizza', 'glass'), ('glass', 'gooseberry'), ('gooseberry', 'Microsoft style latchkey'), ('Microsoft style latchkey', 'fancy refrigerator'), ('fancy refrigerator', 'lavender scented door'), ('lavender scented door', 'insect'), ('rusty table', 'shoddy table'), ('shoddy table', 'shoddy counter'), ('shoddy counter', 'rough shelf'), ('rough shelf', 'spherical portal'),
    ('spherical portal', 'frisbee'), ('frisbee', 'type T passkey'), ('type T passkey', 'type 6 latchkey'), ('type 6 latchkey', 'Canadian style key'), ('Canadian style key', 'rusty rack'), ('rusty rack', 'rough workbench'), ('rough workbench', 'pair of pants'), ('pair of pants', 'rusty table'), ('bug', 'formless keycard'), ('formless keycard', 'non-euclidean safe'), ('non-euclidean safe', 'type A hatch'), ('type A hatch', 'type A latchkey'), ('type A latchkey', "Henderson's style portal"), ("Henderson's style portal", "Henderson's style keycard"),
    ("Henderson's style keycard", 'Microsoft limited edition latchkey'), ('Microsoft limited edition latchkey', 'big refrigerator'), ('big refrigerator', 'Microsoft limited edition chest'), ('Microsoft limited edition chest', 'type W box'), ('type W box', 'type W passkey'), ('type W passkey', 'rectangular chest'), ('rectangular chest', 'bug'), ('American style portal', 'splintery workbench'), ('splintery workbench', 'formless door'), ('formless door', 'lightbulb'), ('lightbulb', 'fresh laundry scented passageway'), ('fresh laundry scented passageway', 'non-euclidean hatch'),
    ('non-euclidean hatch', 'shoddy shelf'), ('shoddy shelf', 'fudge scented gateway'), ('fudge scented gateway', 'fudge scented key'), ('fudge scented key', 'shoddy stand'), ('shoddy stand', 'cuboid latchkey'), ('cuboid latchkey', 'rusty workbench'), ('rusty workbench', 'onion'), ('onion', 'American style portal'), ('toolbox', 'gateway'), ('gateway', 'non-euclidean keycard'), ('non-euclidean keycard', 'keyboard'), ('keyboard', 'formless gate'), ('formless gate', 'non-euclidean door'), ('non-euclidean door', 'non-euclidean portal'),
    ('non-euclidean portal', 'fresh laundry scented passkey'), ('fresh laundry scented passkey', 'Comic Strip Calendar'), ('Comic Strip Calendar', 'toolbox'), ('type H chest', 'type H keycard'), ('type H keycard', 'nest of caterpillars'), ('nest of caterpillars', 'cabbage'), ('cabbage', 'teapot'), ('teapot', 'butterfly'), ('butterfly', 'big fridge'), ('big fridge', 'spork'), ('spork', 'teacup'), ('teacup', 'garlic clove'), ('garlic clove', 'fancy freezer'), ('fancy freezer', 'type H chest'), ('rectangular portal', 'big freezer'), ('big freezer', 'fancy case'),
    ('fancy case', 'American style latchkey'), ('American style latchkey', 'small chest'), ('small chest', 'Microsoft style passkey'), ('Microsoft style passkey', 'blender'), ('blender', 'mug'), ('mug', 'rectangular door'), ('rectangular door', 'lavender scented keycard'), ('lavender scented keycard', 'rectangular portal'), ('bar', 'non-euclidean gate'), ('non-euclidean gate', 'American limited edition key'), ('American limited edition key', 'blanket'), ('blanket', 'Canadian limited edition safe'), ('Canadian limited edition safe', 'Canadian limited edition key'),
    ('Canadian limited edition key', 'bar'), ('stick of butter', 'rusty counter'), ('rusty counter', 'berry'), ('berry', 'rusty stand'), ('rusty stand', 'rough counter'), ('rough counter', 'type 8 passageway'), ('type 8 passageway', 'stick of butter'), ('American limited edition hatch', 'American limited edition passkey'), ('American limited edition passkey', 'durian'), ('durian', 'rough table'), ('rough table', 'spherical passkey'), ('spherical passkey', 'spherical chest'), ('spherical chest', 'neglected locker'), ('neglected locker', 'spherical safe'),
    ('spherical safe', 'sock'), ('sock', 'hat'), ('hat', 'American limited edition hatch'), ('cookie', 'cucumber'), ('cucumber', 'American locker'), ('American locker', 'broccoli'), ('broccoli', 'splintery counter'), ('splintery counter', 'top hat'), ('top hat', 'type O latchkey'), ('type O latchkey', 'Microsoft limited edition hatch'), ('Microsoft limited edition hatch', 'nest of grubs'), ('nest of grubs', 'type Y passkey'), ('type Y passkey', 'shoddy workbench'), ('shoddy workbench', 'cane'), ('cane', 'splintery rack'), ('splintery rack', 'type Q passageway'),
    ('type Q passageway', 'American limited edition safe'), ('American limited edition safe', 'American limited edition latchkey'), ('American limited edition latchkey', 'glove'), ('glove', 'type O portal'), ('type O portal', 'cookie'), ('type Z door', "Henderson's style key"), ("Henderson's style key", "Henderson's style chest"), ("Henderson's style chest", 'TextWorld style portal'), ('TextWorld style portal', 'type Z door'), ('type Z latchkey', 'soap scented latchkey'), ('soap scented latchkey', 'Canadian limited edition passkey'), ('Canadian limited edition passkey', 'type Z latchkey'),
    ('Quote of the Day Calendar', 'formless chest'), ('formless chest', 'book'), ('book', 'gross mantelpiece'), ('gross mantelpiece', 'spherical locker'), ('spherical locker', 'telephone'), ('telephone', 'fly larva'), ('fly larva', 'laptop'), ('laptop', 'formless portal'), ('formless portal', 'Canadian limited edition portal'), ('Canadian limited edition portal', 'Quote of the Day Calendar'), ('American style passkey', 'coconut'), ('coconut', 'fresh laundry scented key'), ('fresh laundry scented key', 'Microsoft limited edition key'), ('Microsoft limited edition key', 'type 3 passageway'),
    ('type 3 passageway', 'shadfly'), ('shadfly', "Henderson's limited edition gate"), ("Henderson's limited edition gate", 'non-euclidean chest'), ('non-euclidean chest', 'cuboid hatch'), ('cuboid hatch', 'type 6 box'), ('type 6 box', 'cake scented safe'), ('cake scented safe', 'cake scented key'), ('cake scented key', 'worn table'), ('worn table', 'American style passkey'), ('loaf of bread', 'rectangular gate'), ('rectangular gate', 'birchwood drawer'), ('birchwood drawer', 'loaf of bread'), ('dusty shelf', 'shiny counter'), ('shiny counter', 'shiny rack'), ('shiny rack', 'shiny board'),
    ('shiny board', 'type 3 box'), ('type 3 box', 'chipped counter'), ('chipped counter', 'cake scented passageway'), ('cake scented passageway', 'dusty shelf'), ('cashew', 'dusty rack'), ('dusty rack', 'non-euclidean latchkey'), ('non-euclidean latchkey', 'TextWorld limited edition gateway'), ('TextWorld limited edition gateway', 'TextWorld limited edition keycard'), ('TextWorld limited edition keycard', 'vanilla scented passageway'), ('vanilla scented passageway', 'cashew'), ("Henderson's style safe", 'gojiberry'), ('gojiberry', "Henderson's hatch"), ("Henderson's hatch", "Henderson's passkey"),
    ("Henderson's passkey", "Henderson's style safe"), ('shiny bench', 'type T gateway'), ('type T gateway', 'melon'), ('melon', 'iron'), ('iron', 'shiny bench'), ("Henderson's limited edition keycard", 'rectangular gateway'), ('rectangular gateway', 'dvd'), ('dvd', 'tablet'), ('tablet', "Henderson's limited edition keycard"), ('player', 'inventory'), ('inventory', 'player')
]
control_pairs_with_rooms_TH = control_pairs_TH + [
    ('pantry', 'steam room'), ('steam room', 'cellar'), ('cellar', 'sauna'), ('sauna', 'cookery'), ('cookery', 'bathroom'), ('bathroom', 'kitchenette'), ('kitchenette', 'launderette'),
    ('launderette', 'parlor'), ('parlor', 'restroom'), ('restroom', 'scullery'), ('scullery', 'workshop'), ('workshop', 'laundromat'), ('laundromat', 'kitchen'), ('kitchen', 'lounge'),
    ('lounge', 'bedroom'), ('bedroom', 'office'), ('office', 'closet'), ('closet', 'dish-pit'), ('dish-pit', 'washroom'), ('washroom', 'playroom'), ('playroom', 'study'),
    ('study', 'basement'), ('basement', 'bedchamber'), ('bedchamber', 'cubicle'), ('cubicle', 'chamber'), ('chamber', 'recreation zone'), ('recreation zone', 'vault'),
    ('vault', 'canteen'), ('canteen', 'studio'), ('studio', 'laundry place'), ('laundry place', 'shower'), ('shower', 'garage'), ('garage', 'salon'), ('salon', 'attic'), ('attic', 'bar'),
    ('bar', 'cookhouse'), ('cookhouse', 'spare room'), ('spare room', 'pantry'),
]
#, ('You', 'inventory'), ('inventory', 'You')]
control_tgt_to_mention_simple = {pair[0]: pair[1] for pair in control_pairs_simple}
control_tgt_to_mention_with_rooms_simple = {pair[0]: pair[1] for pair in control_pairs_with_rooms_simple}
control_mention_to_tgt_simple = {pair[1]: pair[0] for pair in control_pairs_simple}
control_mention_to_tgt_with_rooms_simple = {pair[1]: pair[0] for pair in control_pairs_with_rooms_simple}

control_tgt_to_mention_TH = {pair[0]: pair[1] for pair in control_pairs_TH}
control_tgt_to_mention_with_rooms_TH = {pair[0]: pair[1] for pair in control_pairs_with_rooms_TH}
control_mention_to_tgt_TH = {pair[1]: pair[0] for pair in control_pairs_TH}
control_mention_to_tgt_with_rooms_TH = {pair[1]: pair[0] for pair in control_pairs_with_rooms_TH}