from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_metric import PyRouge
import regex as re
import os
import warnings
import numpy as np
import json

warnings.simplefilter("ignore")
rouge = PyRouge(rouge_n=(1, 2,), rouge_l=True)


def get_bleu(gen_instrs, gt_instrs, line):
    bleu_scores = 0
    valid_instrs = [s.strip(' | ').split(' ') for s in gt_instrs]
    for i, gen_instr in enumerate(gen_instrs):
        length = min(gen_instr.count(' ')+1, 4)
        weights = [1/length for _ in range(length)]
        bleu_score = sentence_bleu(valid_instrs, gen_instr.strip(' | ').split(' '), weights=weights)
        bleu_scores += bleu_score
    return bleu_scores / len(gen_instrs)


def get_rouge_1(gen_instrs, gt_instrs, line):
    rouge = get_rouge(gen_instrs, gt_instrs, line)
    return rouge['rouge-1']

def get_rouge_2(gen_instrs, gt_instrs, line):
    rouge = get_rouge(gen_instrs, gt_instrs, line)
    return rouge['rouge-2']

def get_rouge_l(gen_instrs, gt_instrs, line):
    rouge = get_rouge(gen_instrs, gt_instrs, line)
    return rouge['rouge-l']

def get_rouge(gen_instrs, gt_instrs, line):
    rouge_scores = {}
    for gen_instr in gen_instrs:
        rouge_score = rouge.evaluate([gen_instr], [gt_instrs])
        for metric in rouge_score:
            if metric not in rouge_scores: rouge_scores[metric] = 0
            rouge_scores[metric] += rouge_score[metric]['f']
    return {metric: rouge_scores[metric] / len(gen_instrs) for metric in rouge_scores}

def get_concat_rouge(gen_instrs, gt_instrs, line):
    gen_instrs = sorted(list(set(gen_instrs)))
    gen_instrs = ['\n'.join(gen_instrs)]
    gt_instrs = sorted(list(set(gt_instrs)))
    gt_instrs = ['\n'.join(gt_instrs)]
    concat_rouge = get_rouge(gen_instrs, gt_instrs, line)
    return (concat_rouge['rouge-1'] + concat_rouge['rouge-2'] + concat_rouge['rouge-l']) / 3

def get_prf1(gen_instrs, gt_instrs, line):
    """
    F1 across actions
    """
    f1_scores = 0
    tp_gen_instrs = gen_instrs.intersection(gt_instrs)
    tp = len(tp_gen_instrs)
    npred = len(gen_instrs)
    ngt = len(gt_instrs)
    # assert abs(get_consistency(has_action, actions, gt_actions, has_feedback, feedbacks, gt_feedbacks, lines) * len(seen_lines) - tp) < 1e-5
    p = min(tp / npred, 1)
    r = min(tp / ngt, 1)
    f1 = (2 * p * r) / (p + r) if p + r > 0 else 0
    assert tp <= npred
    return p,r,f1


def get_em(gen_instrs, gt_instrs, line):
    em_scores = [0,0]
    assert len(gt_instrs) == 1
    for gen_instr in gen_instrs:
        em_scores[0] += (gen_instr == gt_instrs[0])
        em_scores[1] += 1
    em_score = em_scores[0] / em_scores[1]
    return em_score


METRICS_TO_EVAL_FN = {
    'BLEU': get_bleu,
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