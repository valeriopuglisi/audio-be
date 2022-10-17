import evaluate
from evaluate import load
import os

datasets_path = "../datasets"
for i, dir in enumerate(os.walk(datasets_path)):
    print(i, dir)
    if i == 1:
        break

print(os.listdir(datasets_path))

from glob import glob


# for path in glob("/path/to/directory/*/", recursive=True):
#     print(path)




cer = load("cer")
wer = load("wer")
predictions = ["hello world", "good night moon"]
references = ["hello world", "good night moon"]
cer_score = cer.compute(predictions=predictions, references=references)
wer_score = wer.compute(predictions=predictions, references=references)
print("cer_score: {}".format(cer_score))
print("wer_score: {}".format(wer_score))
result = {
    "cer_score": cer_score,
    "wer_score": wer_score,
}

params = {"model": "gpt-2"}

evaluate.save(path_or_file="./results/", **result, **params)
