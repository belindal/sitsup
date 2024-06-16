from utils import consistencyCheck, get_em as get_inventory_em
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_metric import PyRouge
import regex as re
import os
import warnings
import numpy as np
import json

warnings.simplefilter("ignore")
rouge = PyRouge(rouge_n=(1, 2,), rouge_l=True)


def get_multiref_bleu(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    multiref_bleu_scores = 0
    if has_action:
        gen_utts = actions
        valid_continuations = [s.split(' | ')[0].strip(' | ').split(' ') for s in gt_actions]
        # multiref_bleu_score = sentence_bleu(valid_actions_tokenized, action.strip().split(' '), weights=weights)
        # multiref_bleu_scores['action'][0] += multiref_bleu_score
        # multiref_bleu_scores['action'][1] += 1
    if has_feedback:
        gen_utts = feedbacks
        valid_continuations = [gt_feedback.split(' | ')[0].strip(' | ').split(' ') for gt_feedback in gt_feedbacks]
        # multiref_bleu_score = sentence_bleu(valid_continuations, feedback.strip().split(' '), weights=weights)
        # multiref_bleu_scores['feedback'][0] += multiref_bleu_score
        # multiref_bleu_scores['feedback'][1] += 1
    assert len(gen_utts) == len(line)
    for i, gen_utt in enumerate(gen_utts):
        length = min(gen_utt.strip(' | ').count(' ')+1, 4)
        weights = [1/length for _ in range(length)]
        multi_bleu_score = sentence_bleu(valid_continuations, gen_utt.strip(' | ').split(' '), weights=weights)
        # try:
        #     assert valid_continuations == line[i]['valid_utts']
        #     assert abs(multi_bleu_score - line[i]['multi_bleu_score']) < 0.00005
        # except:
        #     breakpoint()
        multiref_bleu_scores += multi_bleu_score
    return multiref_bleu_scores / len(gen_utts)


def get_rouge_1(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    rouge = get_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line)
    return rouge['rouge-1']

def get_rouge_2(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    rouge = get_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line)
    return rouge['rouge-2']

def get_rouge_l(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    rouge = get_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line)
    return rouge['rouge-l']

def get_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    rouge_scores = {}
    if has_action:
        gen_utts = actions
        valid_continuations = gt_actions
    if has_feedback:
        gen_utts = feedbacks
        valid_continuations = gt_feedbacks
    for gen_utt in gen_utts:
        rouge_score = rouge.evaluate([gen_utt], [valid_continuations])
        for metric in rouge_score:
            if metric not in rouge_scores: rouge_scores[metric] = 0
            rouge_scores[metric] += rouge_score[metric]['f']
    return {metric: rouge_scores[metric] / len(gen_utts) for metric in rouge_scores}

def get_concat_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    rouge_scores = {}
    # TODO instead of sorting, put together common actions???
    if has_feedback:
        feedbacks = sorted(list(set(feedbacks)))
        feedbacks = ['\n'.join(feedbacks)]
        gt_feedbacks = sorted(list(set(gt_feedbacks)))
        gt_feedbacks = ['\n'.join(gt_feedbacks)]
    if has_action:
        actions = sorted(list(set(actions)))
        actions = ['\n'.join(actions)]
        gt_actions = sorted(list(set(gt_actions)))
        gt_actions = ['\n'.join(gt_actions)]
    concat_rouge = get_rouge(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line)
    return (concat_rouge['rouge-1'] + concat_rouge['rouge-2'] + concat_rouge['rouge-l']) / 3

def get_prf1(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, lines):
    """
    F1 across actions
    """
    f1_scores = 0
    if has_action:
        gen_utts = set(actions)
        valid_continuations = set(gt_actions)
    if has_feedback:
        gen_utts = set(feedbacks)
        valid_continuations = set(gt_feedbacks)
        # assert False
    tp_gen_utts = gen_utts.intersection(valid_continuations)

    """
    num_actually_valid, num_actually_right_feedback = 0, 0
    seen_lines = set()
    tp_gen_utts = set()
    for l, line in enumerate(lines):
        if json.dumps(line) in seen_lines: continue
        seen_lines.add(json.dumps(line))
        is_actually_invalid, _, _, _ = _get_line_consistency(has_action, actions, gt_actions, has_feedback, None, gt_feedbacks, line)
        if not is_actually_invalid:
            if actions[l] not in valid_continuations:
                valid_continuations.add(actions[l])
            tp_gen_utts.add(actions[l])
        # if has_feedback and not is_actually_wrong_feedback:
        #     tp_gen_utts.add(feedbacks[l])
    """
    tp = len(tp_gen_utts)
    npred = len(gen_utts)
    ngt = len(valid_continuations)
    # assert abs(get_consistency(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, lines) * len(seen_lines) - tp) < 1e-5
    p = min(tp / npred, 1)
    r = min(tp / ngt, 1)
    f1 = (2 * p * r) / (p + r) if p + r > 0 else 0
    assert tp <= npred
    return p,r,f1


def get_em(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, line):
    # em_scores = {'action': [], 'feedback': []}
    em_scores = [0,0]
    if has_action:
        for action in actions:
            em_scores[0] += action in gt_actions
            em_scores[1] += 1
    if has_feedback:
        for feedback in feedbacks:
            em_scores[0] += feedback in gt_feedbacks
            em_scores[1] += 1
    # em_scores['all'][0] += (has_action and em_scores['action'][0]) or (has_feedback and em_scores['feedback'][0])
    # em_scores['all'][1] += 1
    return float(em_scores[0]) / em_scores[1]


def get_consistency(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, lines, DEBUG=False):
    num_actually_invalid, num_actually_wrong_feedback = 0, 0
    valid_but_incorrect = {}
    correct_feedback = {}
    if has_feedback:
        assert len(lines) == len(feedbacks)
    if has_action:
        assert len(lines) == len(actions)
    for l, line in enumerate(lines):
        is_actually_invalid, is_actually_wrong_feedback, valid_but_incorrect, correct_feedback = _get_line_consistency(
            has_action, actions, gt_actions, has_feedback, feedbacks[l] if has_feedback else None, gt_feedbacks, line, valid_but_incorrect, correct_feedback)
        if has_action: num_actually_invalid += is_actually_invalid
        if has_feedback: num_actually_wrong_feedback += is_actually_wrong_feedback
    # if len(valid_but_incorrect) > 0:
    #     print({act: len(valid_but_incorrect[act]) for act in valid_but_incorrect})
    # if len(correct_feedback) > 0:
    #     print({feed: len(correct_feedback[feed]) for feed in correct_feedback})
    return 1 - (num_actually_invalid + num_actually_wrong_feedback) / len(lines)


def _get_line_consistency(has_action, actions, gt_actions, has_feedback, feedback, gt_feedbacks, line, valid_but_incorrect=None, correct_feedback=None):
    is_actually_invalid = False
    if has_action and not line['valid?']:
        # is action
        is_actually_invalid = True
        exact_feedback = line['correct_feedback'].strip(" | ").strip()
        actual_correct_feedback = line['correct_feedback']
        if line['correct_feedback'].startswith("You have to") or line['correct_feedback'].startswith("You need to") and line['correct_feedback'].endswith("first."):
            if exact_feedback[len("You have to "):].startswith("unlock"):
                is_actually_invalid = False  # opening before unlocking (phrasing usually unclear about locked or closed)
            actual_correct_feedback = "You have to _ first."
        if line['correct_feedback'].startswith("The") and line['correct_feedback'].endswith("is welded shut."):
            actual_correct_feedback = "The _ is welded shut."
            is_actually_invalid = False
        if line['correct_feedback'].startswith("I only understood you as far as"):
            actual_correct_feedback = "I didn't understand that sentence."
        if line['correct_feedback'].startswith('-='):
            actual_correct_feedback = "Wrong room"
        if line['correct_feedback'] == "That doesn't seem to fit the lock.":
            is_actually_invalid = False
        if line['correct_feedback'] == "That's unlocked at the moment.":
            is_actually_invalid = False
        if line['correct_feedback'] == "That's fixed in place.":
            is_actually_invalid = False
        if line['correct_feedback'].endswith("is wobbly") or line['correct_feedback'].endswith("is stable"):
            actual_correct_feedback = f"The _ is {'wobbly' if line['correct_feedback'].endswith('is wobbly') else 'stable'}"
            is_actually_invalid = False
        if correct_feedback is not None:
            if actual_correct_feedback not in correct_feedback: correct_feedback[actual_correct_feedback] = []
            out = [line['prev_context'], line['gen_utt'], exact_feedback]
            correct_feedback[actual_correct_feedback].append(out)
        # print(line['prev_contexts'])
        # print(line[''])
    is_actually_wrong_feedback = False
    if line['valid?'] and has_feedback and not line['consistent?']:
        # is consequence
        incorrect_feedback_type = None
        is_actually_wrong_feedback = True
        verb = None
        action = actions[2:]
        # try:
        #     assert feedback == line['gen_utt'].strip(" | ")
        # except:
        #     import pdb; pdb.set_trace()
        if action.startswith('go'):
            # seen gt/pred room before, so should know it...
            gt_room_name = line['correct_feedback'].split(' | ')[0]
            pred_room_name = feedback.split(' | ')[0]
            assert pred_room_name != gt_room_name
            seen_room_before = pred_room_name in line['prev_context'] or gt_room_name in line['prev_context']
            if seen_room_before: incorrect_feedback_type = 'incorrect_room'
            else: is_actually_wrong_feedback = False
        elif action.startswith('examine') or action.startswith('look'):
            is_actually_wrong_feedback = False
        elif action.startswith('inventory'):
            try:
                if not get_inventory_em(line['gen_utt'].strip(' | '), line['correct_feedback'].strip(' | '), 'inventory')[0]:
                    incorrect_feedback_type = 'inventory'
                else: is_actually_wrong_feedback = False
            except AssertionError: is_actually_wrong_feedback = True
        elif action.startswith('open') or action.startswith('close') or action.startswith('unlock') or action.startswith('lock'):
            verb = action.split(" ")[0]
            obj = " ".join(action.split(" ")[1:]).split(" with ")[0]
        elif action.startswith('take'):
            verb = "take|pick up"
            obj = " ".join(action.split(" ")[1:]).split(" from ")[0]
        elif action.startswith('put') or action.startswith('drop'):
            verb = action.split(" ")[0]
            obj = " ".join(action.split(" ")[1:]).split(" on ")[0]
        elif action.startswith('insert'):
            verb = "put|insert"
            obj = " ".join(action.split(" ")[1:]).split(" into ")[0]
        elif action.startswith('eat'):
            import pdb; pdb.set_trace()
        else: import pdb; pdb.set_trace()
        if verb:
            if re.match(re.compile(f'You ({verb}) (|the ){obj}'), line['correct_feedback']) or re.match(re.compile(f'You ({verb}) (|the ){obj}'), line['correct_feedback'].split(' | ')[1]):
                incorrect_feedback_type = action.split(' ')[0]
            else: import pdb; pdb.set_trace()
        if incorrect_feedback_type and valid_but_incorrect is not None:
            if incorrect_feedback_type not in valid_but_incorrect: valid_but_incorrect[incorrect_feedback_type] = []
            out = [line['prev_context'], line['gen_utt'], line['gt_utt']]
            valid_but_incorrect[incorrect_feedback_type].append(out)
    return is_actually_invalid, is_actually_wrong_feedback, valid_but_incorrect, correct_feedback


METRICS_TO_EVAL_FN = {
    'CONSISTENCY': get_consistency,
    'MULTIREF_BLEU': get_multiref_bleu,
    # 'BLEU': get_bleu,
    'ROUGE-1': get_rouge_1,
    'ROUGE-2': get_rouge_2,
    'ROUGE-L': get_rouge_l,
    'EM': get_em,
    'F1': get_prf1,
    'CONCAT_ROUGE': get_concat_rouge,
}


def get_loss(train_log_path):
    lm_losses = []
    if os.path.exists(train_log_path):
        with open(train_log_path) as f:
            for line in f:
                if "avg val loss - " in line:
                    line = line.split("avg val loss - ")
                elif "avg val loss: " in line:
                    line = line.split("avg val loss: ")
                else: continue
                lm_losses.append(float(line[-1]))
    return lm_losses