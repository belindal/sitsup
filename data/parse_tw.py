from typing import Iterable

import json
import logging
import os
import pdb
import torch
from tqdm import tqdm
import textworld
from textworld.logic import parser, State
from textworld.logic import Signature, Proposition, Action, Variable, Type
import torch.nn.functional as F
from torch import nn
import itertools
from utils import DEVICE


inv_affecting_actions = ['inventory', 'drop', 'take', 'put', 'insert', 'eat']
state_affecting_actions = inv_affecting_actions + ['open', 'close', 'lock', 'unlock', 'go']
all_actions = state_affecting_actions + ['examine', 'look']
ACTIONS_MAP = {
    'all': all_actions,
    'inv': inv_affecting_actions,
    'state': state_affecting_actions
}

cached_envs = {}

class EntitySet:
    def __init__(self, ent_list: Iterable):
        self.ent_list = sorted(ent_list, key=str)
        self.entity_set = set(ent_list)
        self.has_none = None in self.entity_set
        self.nonNone_ent = self.ent_list[1] if self.ent_list[1] is not None else self.ent_list[0]
    
    def __getitem__(self, i):
        return self.ent_list[i]

    def __hash__(self):
        # order invariant
        set_hash = 0
        for item in self.entity_set:
            set_hash += hash(item)
        return set_hash
    
    def __eq__(self, other):
        return self.entity_set == other.entity_set

    def __str__(self):
        return str(self.entity_set)
    
    def __len__(self):
        return len(self.entity_set)
    
    @staticmethod
    def serialize(entset):
        return json.dumps(entset.ent_list)
    
    @staticmethod
    def deserialize(string: str):
        entity_set = json.loads(string)
        return EntitySet(entity_set)


def convert_fileid_to_gameid(fileid):
    return fileid.split('_')[0]


def translate_inv_items_to_str(inv_items):
    # returns string description of inventory, given a collection of items
    inv_items = list(inv_items)
    if len(inv_items) == 0: return "You are carrying nothing."
    elif len(inv_items) == 1: inv_str = f"You are carrying: {inv_items[0]}."
    elif len(inv_items) >= 2:
        inv_str = f"You are carrying: {', '.join(inv_items[:-1])}"
        inv_str += f' and {inv_items[-1]}.'
    assert len(inv_items) == inv_str.count(', ') + inv_str.count(' and ') + 1
    return inv_str


def translate_inv_str_to_items(inv_str):
    # returns list of items in inventory, given a string description
    if inv_str == 'You are carrying nothing.': return []
    assert inv_str.startswith('You are carrying: ')
    inv_items1 = inv_str.replace('You are carrying: ', '').rstrip('.').split(' and ')
    assert len(inv_items1) <= 2
    inv_items = inv_items1[0].split(', ')
    if len(inv_items1) == 2: inv_items.append(inv_items1[1])
    assert len(inv_items) == inv_str.count(', ') + inv_str.count(' and ') + 1
    return inv_items


def parse_facts_to_nl(facts, inform7_game, get_orig=False):
    # convert list of facts to nl
    nl_facts = []
    orig_facts = []
    nl_facts_set = set()
    for fact in facts:
        # check if already in NL form
        if type(fact) == str: nl_fact = fact
        else:
            fact = Proposition.deserialize(fact)
            nl_fact = inform7_game.gen_source_for_attribute(fact)
            # ensure no repeats
            if nl_fact in nl_facts_set: continue
            if len(nl_fact) == 0:
                # TODO what will we do about these?
                assert fact.name == 'free' or fact.name == 'link'
                continue
        nl_facts.append(nl_fact)
        orig_facts.append(fact)
        nl_facts_set.add(nl_fact)
    if get_orig:
        return nl_facts, orig_facts
    else:
        return nl_facts


import re
def parse_nl_to_facts(nl_facts, game_state, gameid, get_orig=False, cached_templates=None, inform7_game=None):
    facts = []
    invalid_syntax_facts = []
    predicates = game_state['game'].kb.inform7_predicates
    var_names = game_state['game'].kb.inform7_variables
    game_types = game_state['game'].kb.types
    game_ents = game_state.game.infos
    game_type_to_entities = {t: [
        game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)  # game_ents[e].type == t  #
    ] for t in game_types.types}
    entity_name_to_type = {game_ents[e].name: game_ents[e].type for e in game_ents}
    game_types = game_types.types + ["r'"]
    game_type_to_entities["r'"] = game_type_to_entities['r']
    game_type_to_entities['I'] = ['inventory']
    game_type_to_entities['P'] = ['player']
    entity_name_to_type['P'] = 'P'
    entity_name_to_type['I'] = 'I'
    # match fact to predicate
    # over all predicates

    if not cached_templates: cached_templates = {}
    if gameid not in cached_templates:
        cached_templates[gameid] = {}
        entity = re.compile("{"+f"({'|'.join(game_types)})"+"}")
        for signature in predicates:
            nl_template = predicates[signature][1]
            if len(nl_template) == 0: continue
            param_indices = [[m.start(),m.end()] for m in re.finditer(entity, nl_template)]
            symbol_to_type = {param.name: param.type for param in predicates[signature][0].parameters}
            for p in range(len(param_indices)-1,-1,-1):
                param_index_pair = param_indices[p]
                ent_type = symbol_to_type[nl_template[param_index_pair[0]+1:param_index_pair[1]-1]]
                nl_template = f"{nl_template[:param_index_pair[0]]}({'|'.join(game_type_to_entities[ent_type])}){nl_template[param_index_pair[1]:]}"
            regex = re.compile(nl_template)
            param_idx_to_nl_position = []  # param idx in predicate -> position (group #) in nl string
            for param in predicates[signature][0].parameters:
                if param.name != 'I' and param.name != 'P':
                    param_idx = predicates[signature][1].find("{"+param.type+"}")
                    param_idx_to_nl_position.append(param_idx)
            param_idx_to_nl_position = [i[0] for i in sorted(enumerate(param_idx_to_nl_position), key=lambda x:x[1])]
            if regex in cached_templates[gameid]:
                # import pdb; pdb.set_trace()
                assert param_idx_to_nl_position == cached_templates[gameid][regex]['param_idx_to_nl_position']
                assert predicates[signature][0].name == cached_templates[gameid][regex]['predicate'].name
                assert predicates[signature][0].parameters[0].name == cached_templates[gameid][regex]['predicate'].parameters[0].name
                if len(predicates[signature][0].parameters) > 1:
                    assert predicates[signature][0].parameters[1].name == cached_templates[gameid][regex]['predicate'].parameters[1].name
            cached_templates[gameid][regex] = {
                'predicate': predicates[signature][0],
                'param_idx_to_nl_position': param_idx_to_nl_position,
            }
            if "The player carries" in nl_template:
                nl_template = f"T{nl_template.replace('The player carries t', '')} is in the inventory"
                regex = re.compile(nl_template)
                param_idx_to_nl_position = [0]
                cached_templates[gameid][regex] = {
                    'predicate': predicates[signature][0],
                    'param_idx_to_nl_position': param_idx_to_nl_position,
                }

    for nl_fact in nl_facts:
        # regex match template
        for matched_template in cached_templates[gameid]:
            if re.fullmatch(matched_template, nl_fact): break
        try:
            assert re.fullmatch(matched_template, nl_fact)
        except:
            # no template found (invalid syntax)
            invalid_syntax_facts.append(nl_fact)
            continue
        num_nonconstant_params = 0
        mapping = {}  # placeholder -> variable
        param_idx = 0
        for param in cached_templates[gameid][matched_template]['predicate'].parameters:
            if param.name != 'I' and param.name != 'P':
                entity_name = re.search(matched_template, nl_fact).group(cached_templates[gameid][matched_template]['param_idx_to_nl_position'][param_idx]+1)
                try: mapping[param] = Variable(entity_name, entity_name_to_type[entity_name])
                except: import pdb; pdb.set_trace()
                param_idx += 1
            else:
                mapping[param] = Variable(param.name, entity_name_to_type[param.name])
        try:
            fact = Proposition.serialize(cached_templates[gameid][matched_template]['predicate'].instantiate(mapping))
        except:
            import pdb; pdb.set_trace()
        facts.append(fact)
    try: 
        test_nl_facts = set(parse_facts_to_nl(facts, inform7_game, get_orig=False))
        if len(test_nl_facts) == len(facts):
            try: assert test_nl_facts == set(nl_facts) - set(invalid_syntax_facts)
            except AssertionError:
                test_nl_facts = {fact if 'The player carries' not in fact else f'T{fact.replace("The player carries t", "")} is in the inventory' for fact in test_nl_facts}
                assert test_nl_facts == set(nl_facts) - set(invalid_syntax_facts)
    except AssertionError:
        import pdb; pdb.set_trace()
    return facts, cached_templates


def pad_stack(inputs, pad_idx=1, device='cpu'):
    # inputs: ['input_ids', 'attention_mask']: list of tensors, of dim (#facts, seqlen, *)
    input_seqlens = torch.cat([inp['attention_mask'].sum(1) for inp in inputs])
    max_seqlen = input_seqlens.max()
    max_nfacts = max([inp['attention_mask'].size(0) for inp in inputs])
    input_list = []
    mask_list = []
    for i, inp in enumerate(inputs):
        mask_size = list(inp['attention_mask'].size())
        mask_size[1] = max_seqlen - mask_size[1]
        new_mask = torch.cat([inp['attention_mask'], torch.zeros(*mask_size).to(inp['attention_mask'].device, inp['attention_mask'].dtype)], dim=1)
        mask_size[0] =  max_nfacts - mask_size[0]
        new_mask = torch.cat([new_mask, torch.zeros(mask_size[0], max_seqlen, *mask_size[2:]).to(inp['attention_mask'].device, inp['attention_mask'].dtype)], dim=0)
        new_inp = torch.ones(max_nfacts, max_seqlen, *inp['input_ids'].size()[2:]).to(inp['input_ids'].device, inp['input_ids'].dtype) * pad_idx
        # new_inp = torch.ones(inp['attention_mask'].size(0), max_seqlen, *inp['input_ids'].size()[2:]).to(inp['input_ids'].device, inp['input_ids'].dtype) * pad_idx
        new_inp[new_mask.bool()] = inp['input_ids'][inp['attention_mask'].bool()]
        # pad and stack tensors
        mask_list.append(new_mask)
        input_list.append(new_inp)
    return torch.stack(input_list).to(device), torch.stack(mask_list).to(device)

def get_relevant_facts_about(entities, facts, curr_world=None, entity=None, excluded_entities=None, exact_arg_count=True, exact_arg_order=False):
    '''
    entities: list of entities that should *all* appear in list of facts to get (except for the `None` elements)
    excluded_entities: list of entities that should *never* appear in list of facts to get (overrides `entities`)
    exact_arg_count: only get facts with the exact # of non-None arguments as passed-in `entities`
    exact_arg_order: only get facts with the exact order of non-None arguments as passed-in `entities`
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
        # if exact_arg_order, entities must appear in (correct position of) fact
        # otherwise, entities must appear (anywhere) in fact
        for e, entity in enumerate(entities):
            if entity is not None and ((exact_arg_order and entity != fact_argnames[e]) or (not exact_arg_order and entity not in fact_argnames)):
                add_fact = False
                continue
        if add_fact: relevant_facts.append(fact)
    return relevant_facts

def remap_entset(entset, control_mapping):
    """
    Transform entities of entity set according to `control_mapping`
    {ent1, ent2, ...} -> {control_mapping[ent1], control_mapping[ent2], ...}
    """
    remapped_entset = [None for _ in entset]
    # create (possibly control) names of mentions...
    for e, entity in enumerate(entset):
        # if control task, transform mentions...
        entity_name = entity
        if entity is not None:
            entity_name = control_mapping[entity_name] if entity_name in control_mapping else entity_name
        remapped_entset[e] = entity_name
    return remapped_entset

def gen_possible_pairs(data_dir, game_ids):
    print("Getting all entity types")
    # from fact and entity types
    type_to_gid_to_ents = {}
    gameid_to_state = {}
    for game_id in tqdm(game_ids):
        env = textworld.start(os.path.join(data_dir, f'{game_id}.ulx'))
        game_state = env.reset()
        gameid_to_state[game_id] = game_state
        game_ents = game_state.game.infos
        # filter only the types we can store in the inventory...
        game_types = game_state.game.kb.types
        type_to_gid_to_ents[game_id] = {}
        for t in game_types.types:
            if t not in type_to_gid_to_ents[game_id]: type_to_gid_to_ents[game_id][t] = []
            type_to_gid_to_ents[game_id][t] += [game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)]
        type_to_gid_to_ents[game_id]['I'] = ['inventory']
        type_to_gid_to_ents[game_id]['P'] = ['player']

    print("Getting all possible pairs")
    all_possible_pairs = {}
    type_pairs = {}
    for game_id in tqdm(game_ids):
        predicates = gameid_to_state[game_id]['game'].kb.inform7_predicates
        var_names = gameid_to_state[game_id]['game'].kb.inform7_variables
        if game_id not in all_possible_pairs:
            all_possible_pairs[game_id] = set()
            type_pairs[game_id] = set()
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            if signature.types in type_pairs: continue
            type_pairs[game_id].add(signature.types)
            obj_pairs = set(itertools.product(type_to_gid_to_ents[game_id][signature.types[0]], type_to_gid_to_ents[game_id][signature.types[1]])) if len(signature.types) == 2 else set(itertools.product(type_to_gid_to_ents[game_id][signature.types[0]], [None]))
            obj_pairs = list(obj_pairs)
            for p, pair in enumerate(obj_pairs):
                obj_pairs[p] = EntitySet(pair)
            all_possible_pairs[game_id] = all_possible_pairs[game_id].union(obj_pairs)
        all_possible_pairs[game_id] = list(all_possible_pairs[game_id])
    return all_possible_pairs, type_to_gid_to_ents


def gen_negative_tgts(gamefile, state_encoder, probe_outs, tokenizer, game_ids, ent_set_size):
    game_id_to_entities = {}
    game_id_to_objs = {}
    game_ids_to_kb = {}
    for game_id in game_ids:
        if game_id not in game_id_to_entities:
            env = textworld.start(os.path.join(gamefile, f'{game_id}.ulx'))
            game_state = env.reset()
            game_ents = game_state.game.infos
            # filter only the types we can store in the inventory...
            game_types = game_state.game.kb.types
            game_ids_to_kb[game_id] = game_state['game'].kb
            game_id_to_entities[game_id] = {
                t: [game_ents[e].name for e in game_ents if game_types.is_descendant_of(game_ents[e].type, t)] for t in game_types.types
            }
            game_id_to_objs[game_id] = game_id_to_entities[game_id]['o']
    if ent_set_size == 2:
        return gen_all_facts_pairs(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb)
    elif ent_set_size == 1:
        return gen_all_facts_single(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb)
    else:
        raise AssertionError


def gen_all_facts_single(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb):
    fact_to_template = {}
    all_facts = []
    entity_name_to_types = {}
    for game_id in game_ids:
        predicates = game_ids_to_kb[game_id].inform7_predicates
        var_names = game_ids_to_kb[game_id].inform7_variables
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            # skip 3-arg facts (will never be `true` anyway)
            if len(predicates[signature][0].parameters) > 2: continue
            if predicates[signature][1].count('{') > 1:
                # all combinations to fill in the blanks, with at least one 'entity':
                # (r, r, o) -> ('entity', {r0...r5}, {o0...o7}); ({r0...r5}, 'entity', {o0...o7}); ({r0...r5}, {r0...r5}, 'entity');
                # over which blank we fill with `entity`
                blanks_possibles = []
                for b in range(len(signature.types)):
                    blanks_possibles.append([
                        ['entity'] if b2 == b else game_id_to_entities[game_id][typ]
                        for b2, typ in enumerate(signature.types) #if game_ids_to_kb[game_id].types.is_constant(typ)
                    ])
                    for arg_combines in itertools.product(*blanks_possibles[b]):
                        pred_str = predicates[signature][1]
                        for a, arg in enumerate(arg_combines):
                            arg_template_name = predicates[signature][0].parameters[a].name
                            assert "{"+arg_template_name+"}" in pred_str
                            pred_str = pred_str.replace('{'+arg_template_name+'}', arg)
                        fact_to_template[pred_str] = signature
                        all_facts.append(pred_str)
            else:
                pred_str = predicates[signature][1]
                # get first non-constant
                for t, typ in enumerate(signature.types):
                    if not game_ids_to_kb[game_id].types.is_constant(typ): break
                typ_name = predicates[signature][0].parameters[t].name
                assert predicates[signature][0].parameters[t].type == typ
                assert "{"+typ_name+"}" in pred_str
                # pred_str = pred_str.replace("{"+typ_name+"}", var_names[typ])
                pred_str = pred_str.replace("{"+typ_name+"}", "entity")
                fact_to_template[pred_str] = signature
                all_facts.append(pred_str)
            assert pred_str.count('{') == 0
        for typ in game_id_to_entities[game_id]:
            if typ == "I": entity_name_to_types["inventory"] = {"I"}; continue
            if typ == "P": entity_name_to_types["player"] = {"P"}; continue
            for name in game_id_to_entities[game_id][typ]:
                if name is not None:
                    if name not in entity_name_to_types: entity_name_to_types[name] = set()
                    entity_name_to_types[name].add(var_names[typ])
    all_facts = list(set(all_facts))

    if probe_outs is None: probe_outs = {}
    probe_outs['e_name_to_types'] = entity_name_to_types

    '''
    create entity-specific targets
    '''
    # ([bs *] # facts, seqlen): [facts about entity 0, facts about entity 1, etc.]
    probe_outs['all_entity_vectors'] = {}  # fill in entities
    probe_outs['all_entity_inputs'] = {}
    probe_outs['state_to_idx'] = {}
    probe_outs['idx_to_state'] = {}
    for entity in tqdm(entity_name_to_types):
        entset_serialize = EntitySet.serialize(EntitySet([entity]))
        probe_outs['idx_to_state'][entset_serialize] = []
        facts_unique = set()
        for fact in all_facts:
            if "-= " in entity: entity = entity[3:-3].lower()
            old_fact = fact
            fact = fact.replace('entity', entity)
            if fact not in facts_unique:
                probe_outs['idx_to_state'][entset_serialize].append(fact)
                facts_unique.add(fact)
                fact_to_template[fact] = fact_to_template[old_fact]
        probe_outs['idx_to_state'][entset_serialize] = list(set(probe_outs['idx_to_state'][entset_serialize]))
        probe_outs['state_to_idx'][entset_serialize] = {fact: i for i, fact in enumerate(probe_outs['idx_to_state'][entset_serialize])}
        # input_ids: ([bs *] # facts, seqlen), attention_mask: ([bs *] # facts, seqlen)
        probe_outs['all_entity_inputs'][entset_serialize] = tokenizer(probe_outs['idx_to_state'][entset_serialize], return_tensors='pt', padding=True, truncation=True).to(DEVICE)

        encoded_inputs = []
        # save memory
        for split in range(0, probe_outs['all_entity_inputs'][entset_serialize]['input_ids'].size(0),SPLIT_SIZE):
            inp_ids = probe_outs['all_entity_inputs'][entset_serialize]['input_ids'][split:split+SPLIT_SIZE]
            attn_mask = probe_outs['all_entity_inputs'][entset_serialize]['attention_mask'][split:split+SPLIT_SIZE]
            encoded_inputs.append(state_encoder(input_ids=inp_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state.to('cpu'))
        encoded_inputs = torch.cat(encoded_inputs)

        '''
        model forward
        '''
        # encode everything
        # (bs * # facts, seqlen, embeddim)
        probe_outs['all_entity_vectors'][entset_serialize] = {
            'input_ids': encoded_inputs,
            'attention_mask': probe_outs['all_entity_inputs'][entset_serialize]['attention_mask'].to('cpu'),
        }
        probe_outs['all_entity_inputs'][entset_serialize].to('cpu')
    probe_outs['fact_to_template'] = fact_to_template
    return probe_outs


def gen_all_facts_pairs(state_encoder, probe_outs, tokenizer, game_ids, game_id_to_entities, game_ids_to_kb):
    # entity -> all possible facts pertaining to that entity
    # entities: list of batch's entities to probe for...
    # get all containers
    all_facts = []
    entity_name_to_types = {}
    for game_id in game_ids:
        predicates = game_ids_to_kb[game_id].inform7_predicates
        var_names = game_ids_to_kb[game_id].inform7_variables
        # over all predicates
        for signature in predicates:
            if len(predicates[signature][1]) == 0: continue
            pred_str = predicates[signature][1]
            for t, typ in enumerate(signature.types):  # necessary to delete the type-specific templates
                # replace non-constants
                if not game_ids_to_kb[game_id].types.is_constant(typ):
                    typ_name = predicates[signature][0].parameters[t].name
                    assert predicates[signature][0].parameters[t].type == typ
                    assert "{"+typ_name+"}" in pred_str
                    # pred_str = pred_str.replace("{"+typ_name+"}", var_names[typ])
                    pred_str = pred_str.replace("{"+typ_name+"}", f"entity{t}")
            all_facts.append(pred_str)
            assert pred_str.count('{') == 0
        for typ in game_id_to_entities[game_id]:
            if typ == "I": entity_name_to_types["inventory"] = {"I"}; continue
            if typ == "P": entity_name_to_types["player"] = {"P"}; continue
            for name in game_id_to_entities[game_id][typ]:
                if name is not None:
                    if name not in entity_name_to_types: entity_name_to_types[name] = set()
                    if '-=' in name: name = name[3:-3].lower()
                    entity_name_to_types[name].add(var_names[typ])
    all_facts = list(set(all_facts))
    # tokenize
    fact_to_idx = {fact: i for i, fact in enumerate(all_facts)}

    if probe_outs is None: probe_outs = {}
    # probe_outs['state_to_idx'] = fact_to_idx
    probe_outs['idx_to_state'] = all_facts
    probe_outs['e_name_to_types'] = entity_name_to_types

    '''
    create entity-specific targets
    '''
    # ([bs *] # facts, seqlen): [facts about entity 0, facts about entity 1, etc.]
    batch_all_facts = {}  # fill in entities
    token_all_facts = {}
    all_vectors = {}
    fact_to_idx = {}
    # all permutations (orderings)...
    ent_pairs = itertools.combinations([None, *entity_name_to_types], 2)
    if len(entity_name_to_types) > 50: ent_pairs = tqdm(ent_pairs)
    for ent_pair in ent_pairs:
        assert not "I" in ent_pair and not "P" in ent_pair
        # if ent_pair[1] is not None and "-= " in ent_pair[1]: ent_pair[1] = ent_pair[1][3:-3].lower()
        entset = EntitySet(ent_pair)
        entset_serialize = EntitySet.serialize(entset)
        if entset_serialize not in batch_all_facts: batch_all_facts[entset_serialize] = []
        # set invalid facts to `invalid`, to indicate to model not to connect to those
        # (can't delete them straightforwardly as otherwise labels would be unaligned...)
        # TODO test this?? (just aligning labels?)
        for fact in all_facts:
            if 'entity2' in fact: continue
            # TODO
            if entset.has_none:
                # for facts with 2 args, ent_pair cannot have 1 arg
                if 'entity0' in fact and 'entity1' in fact: continue
                # fill whichever is present
                fact_filled = fact.replace('entity0', entset.nonNone_ent).replace('entity1', entset.nonNone_ent)
            else:
                # for facts with 1 arg, ent_pair cannot have 2 args
                if 'entity0' not in fact or 'entity1' not in fact: continue
                # both present
                fact_filled = fact.replace('entity0', entset[0]).replace('entity1', entset[1])
                fact_filled_reverse = fact.replace('entity0', entset[1]).replace('entity1', entset[0])
                batch_all_facts[entset_serialize].append(fact_filled_reverse)
            batch_all_facts[entset_serialize].append(fact_filled)
        fact_to_idx[entset_serialize] = {fact: idx for idx, fact in enumerate(batch_all_facts[entset_serialize])}
        # input_ids: ([bs *] # facts, seqlen), attention_mask: ([bs *] # facts, seqlen)
        token_all_facts[entset_serialize] = tokenizer(batch_all_facts[entset_serialize], return_tensors='pt', padding=True, truncation=True)
        
        tokens = token_all_facts[entset_serialize].to(DEVICE)
        vectors = state_encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], return_dict=True).last_hidden_state.to('cpu')  #OOMs here...

        '''
        model forward
        '''
        # encode everything
        # (bs * # facts, seqlen, embeddim)
        all_vectors[entset_serialize] = {
            'input_ids': vectors,
            'attention_mask': token_all_facts[entset_serialize]['attention_mask'],
        }
    probe_outs['state_to_idx'] = fact_to_idx
    probe_outs['idx_to_state'] = batch_all_facts
    probe_outs['all_entity_vectors'] = all_vectors
    probe_outs['all_entity_inputs'] = token_all_facts

    return probe_outs


ENTITIES_SIMPLE = ['player', 'inventory', 'wooden door', 'chest drawer', 'antique trunk', 'king-size bed', 'old key', 'lettuce', 'tomato plant', 'milk', 'shovel', 'toilet', 'bath', 'sink', 'soap bar', 'toothbrush', 'screen door', 'set of chairs', 'bbq', 'patio table', 'couch', 'low table', 'tv', 'half of a bag of chips', 'remote', 'refrigerator', 'counter', 'stove', 'kitchen island', 'bell pepper', 'apple', 'note']
ENTITIES_TH = [
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
]
ROOMS_SIMPLE = ['garden', 'bathroom', 'kitchen', 'bedroom', 'backyard', 'living room']
control_pairs_simple = [
    ('player', 'inventory'), ('inventory', 'player'), ('wooden door', 'screen door'), ('screen door', 'refrigerator'), ('refrigerator', 'counter'), ('counter', 'stove'),
    ('stove', 'kitchen island'), ('kitchen island', 'apple'), ('apple', 'note'), ('note', 'tomato plant'), ('tomato plant', 'wooden door'), ('bell pepper', 'milk'), ('milk', 'shovel'),
    ('shovel', 'half of a bag of chips'), ('half of a bag of chips', 'bell pepper'), ('toilet', 'bath'), ('bath', 'sink'), ('sink', 'soap bar'), ('soap bar', 'toothbrush'),
    ('toothbrush', 'toilet'), ('lettuce', 'couch'), ('couch', 'low table'), ('low table', 'tv'), ('tv', 'remote'), ('remote', 'lettuce'), ('chest drawer', 'antique trunk'),
    ('antique trunk', 'king-size bed'), ('king-size bed', 'old key'), ('old key', 'chest drawer'), ('set of chairs', 'bbq'), ('bbq', 'patio table'), ('patio table', 'set of chairs')
]
control_pairs_with_rooms_simple = control_pairs_simple + [('garden', 'bathroom'), ('bathroom', 'kitchen'), ('kitchen', 'bedroom'), ('bedroom', 'backyard'), ('backyard', 'living room'), ('living room', 'garden')]

control_tgt_to_mention_simple = {pair[0]: pair[1] for pair in control_pairs_simple}
control_tgt_to_mention_with_rooms_simple = {pair[0]: pair[1] for pair in control_pairs_with_rooms_simple}
control_mention_to_tgt_simple = {pair[1]: pair[0] for pair in control_pairs_simple}
control_mention_to_tgt_with_rooms_simple = {pair[1]: pair[0] for pair in control_pairs_with_rooms_simple}
