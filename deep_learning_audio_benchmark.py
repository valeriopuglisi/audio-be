from deep_learning_audio_features import *
from deep_learning_dict_api import AudioAnalysisAPI
from deep_learning_dict_datasets import Datasets
import evaluate
from evaluate import load
import pandas as pd


def evaluate_metric_on_dataset(task, dataset, model, metric):
    predictions = []
    references = []
    # os.walk(dataset_path)
    test_table = pd.read_table(Datasets[task][dataset]["test_file"])
    # print(test_table.head(10))
    for row in test_table.iterrows():
        audiofile_path = row[1]["path"]
        reference = row[1]["sentence"]
        if audiofile_path.split(".")[-1] != "wav":
            print(audiofile_path)
        prediction = AudioAnalysisAPI[model]['function'](audiofile_path=audiofile_path)
        predictions.append(prediction)
        references.append(reference)
        print("-----------")
        break

    cer = load(metric)
    wer = load("wer")
    cer_score = cer.compute(predictions= predictions, references= references)
    wer_score = wer.compute(predictions= predictions, references= references)
    print("cer_score: {}".format(cer_score))
    print("wer_score: {}".format(wer_score))
    result = {
        "cer_score": cer_score,
        "wer_score": wer_score,
    }
    params = {"model": "gpt-2"}
    evaluate.save(path_or_file="./results/", **result, **params)
    return result


task = "Automatic Speech Recognition"
dataset = "CommonVoice IT"
audio_path = os.path.join("media", "example1.wav")
model = "/api/automatic_speech_recognition/asr_crdnntransformerlm_librispeech_en"
reference = "one two three"

evaluate_metric_on_dataset(
    task=task,
    dataset=dataset,
    model=model,
    metric="cer",
    )
