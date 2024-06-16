import torch
from torch import nn
from tqdm import tqdm
from data import TRIPDataLoader
import json
from utils import DEVICE, score_generations_from_model
from models.models import ContrastiveClassifierHead, JointClassifierHead
from nltk.translate.bleu_score import sentence_bleu
from state_em_utils import gen_states_given_priortxt, score_posttxts_given_states
import warnings
warnings.simplefilter("ignore")



loss_fct = nn.CrossEntropyLoss(reduction='none')

def eval_model(
    args, model, dataloader, tokenizer, eval_type="decoder",
    get_consistency=None, get_bleu=False, get_multi_bleu=False, output_json_fn=None, num_samples=1,
    state_model=None, post_state_model=None, train_state=None, all_cand_output_encodings=None, all_cand_outputs=None,
    objs_can_interact_model=None, dataset=None, post_state_setting=None, ensemble_samples=-1, ensemble_weight=0.5,
):
    model.eval()
    if state_model: state_model.eval()
    if post_state_model: post_state_model.eval()
    gen_em = 0
    gen_f1 = []
    gen_objs_em = 0
    n_consistent = [0,0,0]
    n_total = [0,0,0]
    saved_preds = []
    if type(model) == nn.DataParallel:
        encoder = model.module.get_encoder()
    else:
        encoder = model.get_encoder()
    bleu_scores = 0
    multi_bleu_scores = 0
    multi_bleu_scores_list = []
    num_cand_lang_samples = num_samples
    state_em = []
    state_f1 = []
    if ensemble_samples > -1:
        num_cand_lang_samples = ensemble_samples * num_samples
        assert state_model and post_state_model
        # do ensembling
        num_state_samples = 1
        batch_size = max(40//num_state_samples,1)
        (
            prev_contexts, gen_states_loss, gen_states,
            gt_states, use_gt_state_masks,
        ) = gen_states_given_priortxt(
            state_model=state_model, args=args, dataset=dataset, dataloader=dataloader,
            tokenizer=tokenizer, batchsize=batch_size, train_state=train_state,
            nnegs=0, num_samples=num_state_samples, max_gt_grounded_states=0, post_state_setting=post_state_setting,
        )
    with torch.no_grad():
        tot_val_loss = 0
        n_val = 0
        s_idx = 0
        state_key = getattr(dataloader, 'state_key', train_state[-1] if train_state else None)
        for j, (inputs, lang_tgts, init_state, tgt_state, game_ids, entities) in enumerate(tqdm(dataloader)):
            if inputs is None: continue
            model_inputs = inputs
            if train_state:
                if train_state[0] == "only_fact":
                    model_inputs = tgt_state[state_key+'_input']
                elif train_state[0] == "concat_fact":
                    model_inputs = tgt_state[state_key+'_concat_text']
                elif train_state[0] == "lang_to":
                    lang_tgts = tgt_state[state_key]
            bs = model_inputs['input_ids'].size(0)
            if type(dataloader) == TRIPDataLoader and 'contrastive' in eval_type:
                lang_loss, n_correct, batch_save_preds, n_items = dataloader.evaluate(model, inputs, lang_tgts, tgt_state)
                gen_em += n_correct.item()
                saved_preds += batch_save_preds
                tot_val_loss += lang_loss
                n_val += n_items
                continue
                # TODO ensemble with contrastive????
            elif eval_type == "joint":
                # (bs, numcands,)
                labels = lang_tgts['labels']
                lang_loss, similarity_scores, encoder_hidden = model.loss(inputs=model_inputs, raw_cand_outs={
                    'input_ids': lang_tgts['all_cands_input_ids'], 'attention_mask': lang_tgts['all_cands_attention_mask'],
                }, labels=labels)
            else:
                # NOTE: chooses 1 candidate only
                if "contrastive" in eval_type:
                    # (bs x n_cands, seqlen)
                    cand_input_ids = lang_tgts['all_cands_input_ids'].view(-1,lang_tgts['all_cands_input_ids'].size(-1))
                    cand_attn_mask = lang_tgts['all_cands_attention_mask'].view(-1,lang_tgts['all_cands_attention_mask'].size(-1))
                    if type(model) == ContrastiveClassifierHead:
                        # (bs x n_cands, seqlen)
                        return_dict = model.loss(
                            input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], labels=lang_tgts['labels'],
                            raw_cand_outs={'input_ids': cand_input_ids, 'attention_mask': cand_attn_mask}, return_dict=True,
                        )
                        batch_lang_loss = return_dict['similarity']
                    else:
                        n_repeats = lang_tgts['labels'].size(1)
                        # (bs x n_cands, seqlen)
                        input_ids = model_inputs['input_ids'].repeat_interleave(n_repeats,dim=0)
                        attn_mask = model_inputs['attention_mask'].repeat_interleave(n_repeats,dim=0)
                        # (bs x n_cands, seqlen)
                        return_dict = model(
                            input_ids=input_ids, attention_mask=attn_mask, labels=cand_input_ids, return_dict=True,
                        )
                        batch_lang_loss = loss_fct(return_dict.logits.view(-1, return_dict.logits.size(-1)), cand_input_ids.view(-1))
                        # (bs, n_cands, seqlen) -> (bs, n_cands)
                        batch_lang_loss = batch_lang_loss.view(bs,-1,cand_input_ids.size(-1)).sum(-1)
                    lang_loss = return_dict['loss']
                    # lowest scoring losses per example in batch
                    # (bs,)
                    chosen_tgt = batch_lang_loss.argmin(-1)
                    actual_tgt = (lang_tgts['labels'] == 1).nonzero(as_tuple=False)[:,1]
                    # (bs, 1(#samples), seqlen)
                    generated_next_utt = torch.stack([lang_tgts['all_cands_input_ids'][tgt_idx, chosen_tgt[tgt_idx]] for tgt_idx in range(chosen_tgt.size(0))]).unsqueeze(1)

                else:
                    encoder_outputs = encoder(model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], return_dict=True)
                    # assume bart
                    decoder_inputs = {
                        'input_ids': None,
                        'attention_mask': model_inputs['attention_mask'],
                        'encoder_outputs': encoder_outputs,
                        'labels': lang_tgts['input_ids'],  # automatically generates `decoder_input_ids` out of labels
                        'return_dict': True,
                    }
                    return_dict = model(**decoder_inputs)
                    lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state

            tot_val_loss += lang_loss.mean() * len(model_inputs['input_ids'])
            # save predictions
            # if (output_json_fn or get_consistency is not None or get_bleu or get_multi_bleu):
            # try to avoid generating `examine`...
            # bad_words_ids = [[tokenizer.encode(bad_word, add_prefix_space=True)[1]] for bad_word in ['examine']]
            do_sample = num_cand_lang_samples != 1
            if eval_type == "decoder":
                if torch.cuda.device_count() > 1:
                    generate_fn = model.module.generate
                    pad_token_id = model.module.config.pad_token_id
                else:
                    generate_fn = model.generate
                    pad_token_id = model.config.pad_token_id
                gen_output_dict = generate_fn(
                    input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], max_length=128,
                    decoder_start_token_id=pad_token_id, no_repeat_ngram_size=0, num_beams=1,#num_cand_lang_samples,#
                    # bad_words_ids=bad_words_ids, 
                    do_sample=do_sample,
                    num_return_sequences=num_cand_lang_samples, output_scores=True, return_dict_in_generate=True)
                # (bs x n_lang_samples, seqlen)
                gen_output = gen_output_dict['sequences'][:,1:]
                # (bs, num_lang_samples, seqlen)
                full_generated_next_utt = gen_output.view(-1, num_cand_lang_samples, gen_output.size(-1))
                generated_next_utt = full_generated_next_utt
                if ensemble_samples > -1:
                    # (bs, num_lang_samples); (bs, num_lang_samples, seqlen)
                    gen_output_scores, generated_next_utt = score_generations_from_model(model, tokenizer, gen_output_dict, num_cand_lang_samples)
                    # (bs, num_state_samples, seqlen)
                    batch_gen_states = gen_states[s_idx:s_idx+bs].to('cuda')
                    # (bs, num_lang_samples, seqlen)
                    gen_out_attn_mask = full_generated_next_utt != pad_token_id
                    # (bs, num_lang_samples, n_state_samples)
                    posttxt_scores = []
                    for ls in range(full_generated_next_utt.size(1)):
                        # lang_sample
                        posttxt_scores.append(score_posttxts_given_states(
                            post_state_model, tokenizer, batch_gen_states, lang_tgts['input_ids'],
                            full_generated_next_utt[:,ls,:].unsqueeze(1), gen_out_attn_mask[:,ls,:].unsqueeze(1),
                            batch_pre_inp_ids=model_inputs['input_ids'], batch_pre_attn_mask=model_inputs['attention_mask'],
                        ))
                    # TODO combine states to marginalize over them when getting text' probabilities
                    posttxt_scores = torch.cat(posttxt_scores, 1)
                    posttxt_scores[gen_output_scores == float('inf')] = float('inf')  # filter invalids
                    # (bs, num_lang_samples, n_state_samples)
                    posttxt_scores += gen_states_loss[s_idx:s_idx+bs].unsqueeze(1).repeat(1, num_cand_lang_samples, 1).to(posttxt_scores.device)
                    # (bs, num_lang_samples,)
                    posttxt_scores = posttxt_scores.sum(-1)
                    # normalize before adding
                    gen_output_scores = torch.log_softmax(-gen_output_scores,-1)
                    posttxt_scores = torch.log_softmax(-posttxt_scores,-1)
                    # (bs,) lowest loss = highest probability
                    overall_probs = torch.exp(ensemble_weight * posttxt_scores + (1-ensemble_weight) * gen_output_scores)
                    overall_probs[overall_probs != overall_probs] = 0
                    # rerank lang generations
                    # convert NLLs to probabilities
                    selected_idxs = torch.multinomial(overall_probs, num_samples, replacement=True)
                    generated_next_utt = full_generated_next_utt.gather(1, selected_idxs.unsqueeze(-1).repeat(1,1,full_generated_next_utt.size(-1)))
                    s_idx += bs
            if state_model:
                state_outputs = state_model.generate(
                    input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], max_length=128,
                    decoder_start_token_id=model.config.pad_token_id, no_repeat_ngram_size=0, num_beams=1,#num_cand_lang_samples,#
                    do_sample=do_sample, num_return_sequences=num_cand_lang_samples, output_scores=True, return_dict_in_generate=True,
                )
            if objs_can_interact_model:
                # get objs can interact decoded...
                gen_output = objs_can_interact_model.generate(
                    input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'], max_length=128,
                    decoder_start_token_id=model.config.pad_token_id, no_repeat_ngram_size=0, num_beams=1,
                    # bad_words_ids=bad_words_ids, 
                    do_sample=do_sample, num_return_sequences=num_samples)
                generated_obj_can_interact = gen_output.view(-1, num_samples, gen_output.size(-1))
            for i in range(len(inputs['input_ids'])):
                prev_context = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)
                if train_state and train_state[0] in ["only_fact", "concat_fact"]:
                    model_inputs_str = tokenizer.decode(model_inputs['input_ids'][i], skip_special_tokens=True)
                gt_utt = tokenizer.decode(lang_tgts['input_ids'][i], skip_special_tokens=True)
                is_action = gt_utt.startswith('>')
                if get_multi_bleu:
                    if args.data_type == 'textworld':
                        if is_action:
                            valid_utts = dataset['final_state'][n_val]['valid_actions']
                            valid_utts = ["> "+utt.split(' | ')[0] for utt in valid_utts]
                        else:
                            valid_utts = [dataset['tgts'][n_val].split(' | ')[0]]
                        valid_utts = [utt.strip(' | ').split(' ') for utt in valid_utts]
                    else:
                        valid_utts = [tokenizer.decode(vu, skip_special_tokens=True).split(' ') for vu in lang_tgts['all_cands_input_ids'][i][lang_tgts['labels'][i].bool()]]
                if objs_can_interact_model: gt_objs = tokenizer.decode(tgt_state['objs']['input_ids_can_interact'][i], skip_special_tokens=True)
                if type(model) in [ContrastiveClassifierHead, JointClassifierHead]:
                    sample_indices = [return_dict['similarity'][i].argmax()]
                    num_samples = len(sample_indices)
                if "-=" in gt_utt: gt_utt = gt_utt.split('=- | ')[0] + "=-"
                any_em = False
                any_objs_em = False
                curr_ex_bleu_score = 0
                curr_ex_multi_bleu_score = 0
                for s in range(num_samples):
                    if type(model) in [ContrastiveClassifierHead, JointClassifierHead]:
                        gen_utt = tokenizer.decode(lang_tgts['all_cands_input_ids'][i][sample_indices[s]], skip_special_tokens=True)
                    else:
                        gen_utt = tokenizer.decode(generated_next_utt[i][s], skip_special_tokens=True)
                    # tokenizer??
                    if "-=" in gen_utt: gen_utt = gen_utt.split('=- | ')[0] + "=-"
                    if objs_can_interact_model:
                        # get objs can interact decoded...
                        gen_objs = tokenizer.decode(generated_obj_can_interact[i][s], skip_special_tokens=True)
                    if state_model:
                        state_outputs_str = tokenizer.decode(state_outputs['sequences'][i], skip_special_tokens=True)
                        gt_state_str = tokenizer.decode(tgt_state[state_key]['input_ids'][i], skip_special_tokens=True)
                        output_state = set(state_outputs_str.split("[Next]")[0].strip().split(". "))
                        gt_state = set(gt_state_str.split("[Next]")[0].strip().split(". "))
                        state_em.append(output_state == gt_state)
                        intersection = output_state.intersection(gt_state)
                        if len(intersection) == 0:
                            state_f1.append(0)
                        else:
                            precision = len(intersection) / len(output_state)
                            recall = len(intersection) / len(gt_state)
                            state_f1.append((2 * precision * recall) / (precision + recall))
                    gen_utt_toks = gen_utt.strip(' | ').split(' ')
                    if get_bleu:
                        length = min(len(gen_utt_toks), 4)
                        weights = [1/length for _ in range(length)]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            bleu_score = sentence_bleu([gt_utt.strip(' | ').split(' ')], gen_utt_toks, weights=weights)
                            curr_ex_bleu_score += bleu_score
                    if get_multi_bleu:
                        length = min(len(gen_utt_toks), 4)
                        weights = [1/length for _ in range(length)]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            multi_bleu_score = sentence_bleu(valid_utts, gen_utt_toks, weights=weights)
                            curr_ex_multi_bleu_score += multi_bleu_score
                            multi_bleu_scores_list.append(multi_bleu_score)
                    if get_consistency is not None:
                        consistency_list = get_consistency(args.gamefile, game_ids[i], prev_context, gt_utt, gen_utt, tokenizer, type(model))
                        
                        for v_idx in range(len(n_consistent)):
                            if consistency_list[v_idx] is not None:
                                n_consistent[v_idx] += consistency_list[v_idx]
                                n_total[v_idx] += 1
                    if output_json_fn is not None:
                        saved_pred = {
                            'prev_context': prev_context, 'gt_utt': gt_utt, 'gen_utt': gen_utt, 'game_id': game_ids[i],
                        }
                        if train_state and train_state[0] in ['only_fact', 'concat_fact']:
                            saved_pred['model_inputs'] = model_inputs_str
                        elif train_state and train_state[0] == 'LM_aux':
                            saved_pred['model_gen_state'] = state_outputs_str
                            saved_pred['aux_gt_state'] = gt_state_str
                        if get_consistency:
                            saved_pred['correct_feedback'] = consistency_list[3]
                            saved_pred['consistent?'] = consistency_list[0]
                            saved_pred['valid?'] = not consistency_list[1]
                        if objs_can_interact_model:
                            saved_pred['gen_objs'] = gen_objs
                            saved_pred['gt_objs'] = gt_objs
                            gen_objs = set(gen_objs[len("You can interact with "):].strip().split(', '))
                            gt_objs = set(gt_objs[len("You can interact with "):].strip().split(', '))
                        if get_multi_bleu:
                            saved_pred['multi_bleu_score'] = multi_bleu_score
                            saved_pred['valid_utts'] = valid_utts
                        if ensemble_samples > -1:
                            all_ensemble_utts = [tokenizer.decode(full_generated_next_utt[i,samp_num,:], skip_special_tokens=True) for samp_num in range(num_cand_lang_samples)]
                            gen_state_nl = [tokenizer.decode(batch_gen_states[i,samp_num,:], skip_special_tokens=True) for samp_num in range(num_state_samples)]
                            saved_pred['ensemble_cand_utts'] = all_ensemble_utts
                            saved_pred['gen_states'] = gen_state_nl
                        saved_preds.append(saved_pred)
                    any_em |= (gen_utt == gt_utt)
                    if objs_can_interact_model: any_objs_em |= (gen_objs == gt_objs)
                    if train_state and train_state[0] == "lang_to":
                        if args.data_type == "trip":
                            gen_states = gen_utt.split("[Next]")[0].strip()
                            gt_states = gt_utt.split("[Next]")[0].strip()
                            gen_states = set(gen_states.split('. '))
                            gt_states = set(gt_states.split('. '))
                        if len(gen_states.intersection(gt_states)) == 0:
                            gen_f1.append(0)
                        else:
                            precision = len(gen_states.intersection(gt_states)) / len(gen_states)
                            recall = len(gen_states.intersection(gt_states)) / len(gt_states)
                            gen_f1.append(2 * precision * recall / (precision + recall))
                bleu_scores += curr_ex_bleu_score / num_samples if num_samples > 0 else 0
                multi_bleu_scores += curr_ex_multi_bleu_score / num_samples if num_samples > 0 else 0

                gen_em += any_em
                if objs_can_interact_model: gen_objs_em += any_objs_em
                n_val += 1
    if output_json_fn is not None:
        with open(output_json_fn, 'w') as wf:
            for pred in saved_preds:
                wf.write(json.dumps(pred) + '\n')
        print(f"Saved model prediction to {output_json_fn}")

    print("n_val", n_val)
    print("EM with gt utterance: ", gen_em / n_val)
    other_metrics = {}
    if train_state and train_state[0] == "lang_to":
        print("State F1: ", sum(gen_f1) / len(gen_f1))
        other_metrics["state_f1"] = sum(gen_f1) / len(gen_f1)
    if len(state_em) > 0:
        print("State EM: ", sum(state_em) / len(state_em))
        print("State F1: ", sum(state_f1) / len(state_f1))
    if objs_can_interact_model: print("EM with gt can-interact objects: ", gen_objs_em / n_val)
    avg_val_loss = tot_val_loss.mean() / n_val
    p_consistent = [n_consistent[c] / n_total[c] if n_total[c] > 0 else 0 for c in range(len(n_consistent))]
    bleu_scores = bleu_scores / n_val
    multi_bleu_scores = multi_bleu_scores / n_val
    return n_val, avg_val_loss, p_consistent, bleu_scores, multi_bleu_scores, gen_em, other_metrics


def get_all_outputs(dev_dataset, model=None, tokenizer=None):
    all_actions = set()
    for data in dev_dataset['final_state']:
        all_actions = all_actions.union(set(data['valid_actions']))
        all_actions = all_actions.union(set(data['invalid_actions']))
    all_actions = list(all_actions)
    all_actions = ["> "+action.split(' | ')[0] for action in all_actions]
    bs = 64
    if model:
        model.eval()
        encoded_all_actions = []
        for i in tqdm(range(0, len(all_actions), bs)):
            tokenized_action = tokenizer(all_actions[i:min(i+bs, len(all_actions))], return_tensors='pt', padding=True, truncation=False).to(DEVICE)
            encoded_action = model.out_encoder(input_ids=tokenized_action['input_ids'], attention_mask=tokenized_action['attention_mask'], return_dict=True)
            encoded_action = model.mean(encoded_action.last_hidden_state, tokenized_action['attention_mask'], 1).detach().to('cpu')
            encoded_all_actions.append(encoded_action)
        encoded_all_actions = torch.cat(encoded_all_actions)
        return all_actions, encoded_all_actions
