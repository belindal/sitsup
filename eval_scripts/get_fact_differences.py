import json


gt_facts_file = "gpt3-text-davinci-002/textworld/with_latent_origstate_5cands_outs.jsonl"
latent_facts_file = "gpt3-text-davinci-002/textworld/with_latent_origstate_5cands_prompt.jsonl"


def parse_facts(filename):
    ctxt_to_facts = {}
    ctxt_to_facts_cands = {}
    curr_ctxt = []
    with open(filename) as f:
        for line in f:
            if "TRIP" in filename and line.strip() in ["OK", "Not OK"]:
                continue
            if line.strip() == "":
                curr_ctxt = []
                continue
            if line.startswith("Known facts"):
                ctxt_to_facts["\n".join(curr_ctxt)] = line[len("Known facts: "):].strip().split(". ")
            elif line.startswith("Candidates"):
                ctxt_to_facts_cands["\n".join(curr_ctxt)] = line[len("Candidates: "):].strip().split(". ")
            else:
                curr_ctxt.append(line.strip())
    return ctxt_to_facts, ctxt_to_facts_cands

gt_ctxt_to_facts, _ = parse_facts(gt_facts_file)
latent_ctxt_to_facts, latent_ctxt_to_facts_cands = parse_facts(latent_facts_file)

for ctxt in gt_ctxt_to_facts:
    breakpoint()
    fact_diff = set(gt_ctxt_to_facts[ctxt]).symmetric_difference(set(latent_ctxt_to_facts[ctxt]))