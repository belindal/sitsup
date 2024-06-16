import torch
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader, Dataset, IterableDataset
from utils import DEVICE
import glob
import random


# def convert_entity_state_to_nl(entity_state):
#     return f"{entity_state['attr']} of {entity_state['entity']} is {entity_state['before']}"

# def convert_nl_to_entity_state(entity_state_nl):
#     entity_state = {}
#     if 'was before' in entity_state_nl and 'afterwards' in entity_state_nl:
#         return
#     else:
#         return 

# def merge_states():


class OpenPIDataset(Dataset):
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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        self.data[i]['idx'] = i
        return self.data[i]
    
    def load_state(self, state): #get_before_state: bool = False, get_after_state: bool = True, get_change_in_state: bool = True):
        # Load a single state
        """
        state: dict containing (unparsed) state info
        Returns: dict containing `before` state, `after` state, and `change` in state
        """
        state_info_to_ret = {'before': set(), 'after': set(), 'change': set()}
        entities = set()
        for entity_idx in range(len(state['answers_metadata'])):
            entity_state = state['answers_metadata'][entity_idx]
            entities.add(entity_state['entity'])
            try:
                answer_words = entity_state['answer'].lower().strip().replace(',', '').replace('.', '').split()
                answer2_words = f"{entity_state['attr']} of {entity_state['entity']} was {entity_state['before']} before and {entity_state['after']} afterwards".lower().strip().replace(',', '').replace('.', '').split()
                answer_articles_removed = ' '.join([word for word in answer_words if word not in ['a', 'an', 'the', 'your', 'my', 'their', 'other', 'now']])
                answer2_articles_removed = ' '.join([word for word in answer2_words if word not in ['a', 'an', 'the', 'your', 'my', 'their', 'other', 'now']])
                assert answer_articles_removed == answer2_articles_removed
            except AssertionError:
                print(answer_articles_removed)
                print(answer2_articles_removed)
                # nl_answer_attr = entity_state['answer'].split(' of ')[0].lower()
                # nl_answer_entity = entity_state['answer'].split(' of ')[1].split(' was ')[0].lower()
                # nl_before_state = entity_state['answer'].split(' was ')[1].split(' before and ')[0].lower()
                # nl_after_state = entity_state['answer'].split(' before and ')[1].split(' afterwards')[0].lower()
                # check_attr = nl_answer_attr == entity_state['attr'].lower()
                # check_entity = nl_answer_entity == entity_state['entity'].lower()
                # check_before_state = nl_before_state == entity_state['before'].lower()
                # check_after_state = nl_after_state == entity_state['after'].lower()
                # import pdb; pdb.set_trace()
            # taking *before* the query (query is next completion)
            state_info_to_ret['before'].add(f"{entity_state['attr']} of {entity_state['entity']} is {entity_state['before']}")
            state_info_to_ret['after'].add(f"{entity_state['attr']} of {entity_state['entity']} is {entity_state['after']}")
            # TODO use which version???
            state_info_to_ret['change'].add(f"{entity_state['attr']} of {entity_state['entity']} was {entity_state['before']} before and {entity_state['after']} afterwards")
        return state_info_to_ret, entities

    def merge_states_by_attr_entity(self, state_sets, keep_first=True):
        """
        Returns union over `state_sets`, looking only at entity identity and attribute
        Assumes described state/state change is the same given entity and attribute
        """
        ent_attr2state = {}
        for state_set in state_sets:
            for state in state_set:
                if ' is ' in state:
                    ent_attr = state.split(' is ')[0]
                elif ' was ' in staet:
                    ent_attr = state.split(' was ')[0]
                if keep_first:
                    if ent_attr not in ent_attr2state:
                        ent_attr2state[ent_attr] = state
                else:
                    # keep last
                    ent_attr2state[ent_attr] = state
        return set(ent_attr2state.values())

    def load_data(self, data_dir, data_split, max_seq_len, max_data_size, max_num_aligned):
        num_aligned = 0
        num_data = 0
        context_fp = os.path.join(os.path.join(data_dir, data_split), "id_question_metadata.jsonl")
        state_fp = os.path.join(os.path.join(data_dir, data_split), "id_answers_metadata.jsonl")
        states = open(state_fp).readlines()
        contexts = open(context_fp).readlines()
        print("Finished reading context and state files")

        task2instrs = {}  # task to instructions
        # init_state = json.loads(states[0])
        # # TODO Take union over initial states at each timestep (not necessarily--initial state described might correspond to a step)
        # init_state = ', '.join(list(self.load_state(init_state)['before']))
        prev_state = None
        for c, ctxt in tqdm(enumerate(contexts)):
            if num_data >= max_data_size:
                break
            
            ctxt = json.loads(ctxt)
            # invalid
            if "context" not in ctxt["question_metadata"]:
                continue
            state = json.loads(states[c])

            # get id, task, instruction number
            cid = ctxt['id']
            assert ctxt['id'] == state['id']
            assert cid.startswith("www.wikihow.com/")
            task_name = cid[len("www.wikihow.com/"):].split("||")[0].replace("-", " ").lower()
            instr_num = int(cid.split("||")[1])

            # get state
            state, entities = self.load_state(state)
            curr_state = state['before']
            if instr_num == 1:
                prev_state = None
                init_state = ', '.join(list(curr_state))
            else:
                curr_state = self.merge_states_by_attr_entity([curr_state, prev_state['after']])
            # TODO CHECK UNION???
            all_entity_states = {
                'all_curr_state_facts': list(curr_state),
                'all_state_change_facts': list(state['change']),
            }
            if num_aligned < max_num_aligned:
                all_entity_states['curr_state_facts'] = all_entity_states['all_curr_state_facts']
                all_entity_states['state_change_facts'] = all_entity_states['all_state_change_facts']
                num_aligned += 1
            else:
                all_entity_states['curr_state_facts'] = None
                all_entity_states['state_change_facts'] = None
            prev_state = state

            # make context
            # TODO should we add the initial state?
            instr = [f'{init_state} [SEP] How to {task_name}.']  #[f'{init_state} [SEP] How to {task_name}.']
            if len(ctxt["question_metadata"]["context"]) > 0:
                instr.append(ctxt["question_metadata"]["context"])
            instr = ' | '.join(instr)
            instr_tokens = self.tokenizer.tokenize(instr)
            if len(instr_tokens) > max_seq_len:
                continue

            if self.control_input:
                context = 'Entities include ' + ', '.join(entities)
            else:
                context = instr
            if task_name not in task2instrs:
                task2instrs[task_name] = []
            task2instrs[task_name].append({
                'id': cid,
                'context': context,
                'next_instr': ctxt["question_metadata"]["query"],
                'post_context': f'{ctxt["question_metadata"]["query"]} {ctxt["question_metadata"]["future_context"]}',
                'states': all_entity_states,
            })
            # check index of data == instr_num
            assert len(task2instrs[task_name]) == instr_num

            num_data += 1
        # shuffle contexts
        tasks_order = list(task2instrs.keys())
        if self.randseed:
            random.seed(self.randseed)
            random.shuffle(tasks_order)
        
        # linearize data
        full_data = []
        for task in tasks_order:
            for instr_entry in task2instrs[task]:
                full_data.append(instr_entry)

        return full_data
    
    def set_expected_states(self, expected_states):
        import pdb; pdb.set_trace()

        for eidx, entry in enumerate(self.data):
            assert expected_states[eidx]['id'] == entry['id']
            self.data[eidx]['states'] = expected_states[eidx]
        # TODO


class OpenPIDataLoader(DataLoader):
    def __init__(self, dataset: OpenPIDataset, tokenizer, batch_size, train_state=None, contrastive: bool=False):
        """
        state_keys_to_get: [(init/final_state, key); (init/final_state, key)]
        """
        super().__init__(dataset, batch_size, collate_fn=self.collate_fn)
        self.tokenizer = tokenizer
        # self.dataset = dataset
        self.train_state = train_state
        self.state_key = None
        if self.train_state:
            self.state_key = self.train_state[-1]
        self.contrastive = 'contrastive' in contrastive
    
    def collate_fn(self, batch):
        batch_ctxts = self.tokenizer([ex['context'] for ex in batch], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        batch_tgts = self.tokenizer([ex['next_instr'] for ex in batch], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        batch_ids = [ex['id'] for ex in batch]
        if self.state_key:
            batch_tgt_facts = [
                ', '.join([
                    state for state in ex['states'][self.state_key]
                ]) if ex['states'][self.state_key] is not None else '' for ex in batch
            ]
            batch_all_tgt_facts = [
                ', '.join([
                    state for state in ex['states'][f'all_{self.state_key}']
                ]) if ex['states'][f'all_{self.state_key}'] is not None else '' for ex in batch
            ]
            batch_concat_text = [
                f'{ex["context"]} [SEP] {batch_tgt_facts[i]}' if len(batch_tgt_facts) > 0 else ex['context'] for i, ex in enumerate(batch)
            ]
            batch_tgt_state_tok = self.tokenizer(batch_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_all_tgt_state_tok = self.tokenizer(batch_all_tgt_facts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_concat_text_tok = self.tokenizer(batch_concat_text, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            batch_tgt_state_ret = {
                self.state_key: batch_tgt_state_tok,
                self.state_key + '_input': batch_tgt_state_tok,
                self.state_key + '_gold': batch_all_tgt_state_tok,
                self.state_key + '_concat_text': batch_concat_text_tok,
            }
        else:
            batch_tgt_state_ret = {}

        if self.contrastive:
            # (bs, n_cands, seqlen)
            batch_tgts['all_cands_input_ids'] = batch_tgts['input_ids'].unsqueeze(0).repeat(batch_tgts['input_ids'].size(0),1,1)
            batch_tgts['all_cands_attention_mask'] = batch_tgts['attention_mask'].unsqueeze(0).repeat(batch_tgts['attention_mask'].size(0),1,1)
            # (bs, n_cands,)
            batch_tgts['labels'] = torch.eye(len(batch)).to(DEVICE)
        # (inputs, lang_tgts, init_state, tgt_state, game_ids, entities)
        return batch_ctxts, batch_tgts, None, batch_tgt_state_ret, batch_ids, None


get_openpi_consistency = None
