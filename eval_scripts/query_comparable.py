import json
from metrics.tw_metrics import get_consistency
# import faiss

gpt3_model = "text-davinci-002"

def get_correctness(fact_file, story_lengths=None):
    stories = {}
    n_correct = 0
    n_total = 0
    precision = []
    recall = []
    f1 = []
    longest_story_length = 0
    currstory = None
    with open(fact_file) as f:
        for line in f:
            if line == "\n":
                continue
            line = line.strip()
            if line.startswith(f"{len(stories)}."):
                currstory = line[len(f"{len(stories)}."):].strip()
                if story_lengths is None or currstory in story_lengths:
                    stories[currstory] = []
                    # .append([])
            if story_lengths is not None and currstory in story_lengths and len(stories[currstory]) > len(story_lengths[currstory]): continue
            if "textworld" in fact_file and line.startswith("Generation ("):
                # if '> unlock' in line: continue
                pred_actions = []
                while line.startswith("Generation ("):
                    action = ": ".join(line.split(": ")[1:]).strip()
                    pred_actions.append(action)
                    line = f.readline()
                valid_actions = json.loads(line[len("All valid actions: "):].strip())
                invalid_actions = json.loads(f.readline()[len("All invalid actions: "):].strip())
                # correct = 0.0
                correct = False
                for action in pred_actions:
                    # correct += action in valid_actions
                    correct |= action in valid_actions
                # correct /= len(pred_actions)
                p = len(set(valid_actions).intersection(set(pred_actions))) / len(set(pred_actions))
                r = len(set(valid_actions).intersection(set(pred_actions))) / min(len(set(valid_actions)), 5)
                precision.append(p)
                recall.append(r)
                f1.append(0 if p + r == 0 else 2 * p * r / (p + r))
                # correct = line[len("Generation ("):len("Generation (")+1] == ")"
            elif "TRIP" in fact_file and (line.startswith("OK") or line.startswith("Not OK")):
                # breakpoint()
                # pred = "OK" if line[:2] == "OK" else "Not OK"
                correct = not line.endswith("X")
            else:
                continue
            if story_lengths is None or currstory in story_lengths:
                n_correct += correct
                n_total += 1
                stories[currstory].append(correct)
                longest_story_length = max(longest_story_length, len(stories[currstory]))

    full_story_accuracy = 0
    n_total_stories = 0
    for story in stories:
        assert story_lengths is None or story in story_lengths
        full_story_accuracy += sum(stories[story]) == len(stories[story])
        n_total_stories += 1

    if n_total == 0:
        breakpoint()
    print(longest_story_length)
    print(f"Correctness: {n_correct / n_total}")
    print(f"Correctness (Storywise): {full_story_accuracy} // {full_story_accuracy / n_total_stories}")
    if len(recall) > 0:
        print(f"Avg. Precision (Macro-average): {sum(precision) / len(precision)}")
        print(f"Avg. Recall (Macro-average): {sum(recall) / len(recall)}")
        print(f"Avg. F1 (Macro-average): {sum(f1) / len(f1)}")
    return stories

# withfact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP_old/with_origstate_outs.jsonl")
withfact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP/with_ourstate_outs_16.jsonl")
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP/with_latent_ourstate_5cands_outs.jsonl", withfact_stories)
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP/with_latent_ourstate_1cands_outs.jsonl", withfact_stories)
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP/with_latent_ourstate_0cands_outs.jsonl", withfact_stories)
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP_old/with_latent_ourstate_5cands_outs.jsonl", withfact_stories)
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP_old/with_latent_ourstate_1cands_outs.jsonl", withfact_stories)
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/TRIP_old/with_latent_ourstate_0cands_outs.jsonl", withfact_stories)
nofact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP/classify_textonly_comparable_outs1.jsonl", withfact_stories)
nofact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP/textonly_outs.jsonl", withfact_stories)


# withorigfact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP/classify_with_origstate_outs.jsonl")
# withorigfact_stories = get_correctness(f"gpt3-{gpt3_model}/TRIP/classify_with_ourstate_outs.jsonl", withorigfact_stories)

# story_accuracies = [[],[]]
# for story in withfact_stories:
#     withfact_story_accuracy = sum(withfact_stories[story]) == len(withfact_stories[story])
#     nofact_story_accuracy = sum(nofact_stories[story]) == len(nofact_stories[story])
#     story_accuracies[0].append(withfact_story_accuracy)
#     story_accuracies[1].append(nofact_story_accuracy)
# for story in story_accuracies:
#     print(sum(story) / len(story))

"""
# withfact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/with_origstate_outs.jsonl")
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_0cands_outs.jsonl", withfact_stories)
# withfact_stories_LS1 = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_1cands_outs.jsonl", withfact_stories)
# withfact_stories_LS5 = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_5cands_outs.jsonl", withfact_stories)
# nofact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/textonly_outs.jsonl", withfact_stories)
# gpt3-text-davinci-002/textworld/with_latent_origstate_1cands_outs1.jsonl
withfact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_with_origstate_outs5.jsonl")
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_5cands_outs5.jsonl", withfact_stories)
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_1cands_outs5.jsonl", withfact_stories)
withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_0cands_outs5.jsonl", withfact_stories)
# nofact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_prompt_with_Falsestates_comparable_out_with_Falsestates1.jsonl", withfact_stories)
# nofact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_prompt_with_origstates_out_with_Falsestates5.jsonl", withfact_stories)
nofact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_textonly_outs5.jsonl", withfact_stories)
"""

withfact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_prompt_with_origstates_out_with_origstates1.jsonl")
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_5cands_outs.jsonl", withfact_stories)
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_1cands_outs.jsonl", withfact_stories)
# withfact_stories_less_annots = get_correctness(f"gpt3-{gpt3_model}/textworld/with_latent_origstate_0cands_outs1.jsonl", withfact_stories)
nofact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_textonly_comparable_outs1.jsonl", withfact_stories)
# withfact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_prompt_with_Falsestates_out_with_Falsestates1.jsonl", withfact_stories)

withfact_stories = get_correctness(f"gpt3-{gpt3_model}/textworld/generate_prompt_with_Falsestates_comparable_out_with_Falsestates5.jsonl", withfact_stories)
"""
nofact_stories = {}
story_idx = 0
n_correct = 0
n_total = 0
currstory = None
with open(withoutfacts) as f:
    for line in f:
        if line == "\n":
            continue
        line = line.strip()
        if line.startswith(f"{len(nofact_stories)}."):
            currstory = line[len(f"{len(nofact_stories)}."):].strip()  #line.split(f"{len(nofact_stories)}.")[-1]
            nofact_stories[currstory] = []
            # story_idx = len(nofact_stories) - 1
        if len(withfact_stories[currstory]) <= len(nofact_stories[currstory]): continue
        if line.startswith("Generation ("):
            if '> unlock' in line: continue
            action = ": ".join(line.split(": ")[1:])
            valid_actions = json.loads(f.readline()[len("All valid actions: "):].strip())
            invalid_actions = json.loads(f.readline()[len("All invalid actions: "):].strip())
            correct = action in valid_actions
            nofact_stories[currstory].append(correct)
            n_correct += correct
            n_total += 1
print(f"Correctness: {n_correct / n_total}")
"""
"""
storywise_accuracy = []
for currstory in withfact_stories:
    print(currstory)
    print(sum(withfact_stories[currstory]) / len(withfact_stories[currstory]))
    print(sum(nofact_stories[currstory]) / len(nofact_stories[currstory]))
breakpoint()
# """

