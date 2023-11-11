import random
import api.util as util

formats = ["{instruction_prompt}"]

data = util.load_json("data/alpaca_quality/comparison_data_v2.json")
medium_quality = []
high_quality = []
reason_1 = 0
reason_2 = 0
for example in data:
    if len(example["responses_and_scores"]) < 3:
        reason_1 += 1
        continue
    responses = example["responses_and_scores"]
    medium = [r for r in example["responses_and_scores"] if r["score"] in [4, 5]]
    good = [r for r in example["responses_and_scores"] if r["score"] in [9, 10]]

    if len(medium) != 0 and len(good) != 0:
        if random.choice([0,1]) == 0:
            high_quality.append({
                "instruction_prompt": example["user_input"],
                "completions": [medium[0]["response"], good[0]["response"]],
                "preferred_completion": good[0]["response"],
                "best_score": good[0]["score"],
                "worse_score": medium[0]["score"],
            })
        else:
            medium_quality.append({
                "instruction_prompt": example["user_input"],
                "medium_response": medium[0]["response"],
                "medium_quality_score": medium[0]["score"],
            })

print(len(medium_quality))
print(len(high_quality))
for e in medium_quality:
    bad_response = random.choice(alpaca)["completions"][2]
    e["completions"] = [e["medium_response"], bad_response]
    e["preferred_completion"] = e["medium_response"]
    e["best_score"] = e["medium_quality_score"]
    del e["medium_response"]
    del e["medium_quality_score"]

high_quality_data = {
    "name": "alpaca_high_quality",
    "formats": formats,
    "examples": high_quality,
}
util.save_json(high_quality_data, "data/alpaca_high_quality/wrangled.json")

low_quality_data = {
    "name": "alpaca_low_quality",
    "formats": formats,
    "examples": medium_quality,
}

util.save_json(low_quality_data, "data/alpaca_low_quality/wrangled.json")