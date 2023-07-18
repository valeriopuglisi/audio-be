# %%
from deep_learning_features_audio import *
from deep_learning_dict_api import AudioAnalysisAPI
from deep_learning_dict_datasets import Datasets
import evaluate
from evaluate import load
import pandas as pd
from pathlib import Path

# %%
def asr_evaluate_metric_with_model_on_commonvoice(task, dataset, model, metrics, n_test):
    test_done = 0
    errors = 0
    predictions = []
    references = []
    result = {
        "evaluation": {}
    }
    # os.walk(dataset_path)
    test_table = pd.read_table(Datasets[task][dataset]["test_file"])
    test_audio_path = Datasets[task][dataset]["path"]
    tot_sample = test_table.shape[0]
    if n_test is None:
        n_test = tot_sample
    for i, row in enumerate(test_table.iterrows()):
        print("Benchmarking: {}/{}".format(i, test_table.shape[0]))
        audiofile_path = row[1]["path"]
        wav_audiofile_path = os.path.splitext(audiofile_path)[0] + '.wav'
        reference = row[1]["sentence"]
        print("reference:{}".format(reference))
        try:
            audio_path = os.path.join(test_audio_path, "wavs", wav_audiofile_path)
            if os.path.isfile(audio_path):
                prediction = AudioAnalysisAPI[model]['function'](audiofile_path=audio_path)
                predictions.append(prediction.lower())
                references.append(reference.lower())
                print("audiofile_path: {}".format(audio_path))
                print("reference: {}".format(reference.lower()))
                print("prediction:{}\n".format(prediction.lower()))
                test_done += 1
            else:
                print(audio_path, " file doesn't exist")
                errors+=1
            if i == n_test:
                break
        except Exception as e:
            print(e)
            errors += 1
            pass
    for metric in metrics:
        loaded_metric = load(metric)
        # wer = load("wer")
        caluculated_metric = loaded_metric.compute(predictions=predictions, references=references)
        # wer_score = wer.compute(predictions=predictions, references=references)
        print("{}: {}".format(metric, caluculated_metric))
        # print("wer_score: {}".format(wer_score))
        result['evaluation'][metric] = caluculated_metric
    params = {"model": model,
            "dataset": dataset,
            "n_test": n_test,
            "test_done": test_done,
            "errors": errors,
            "tot_sample": tot_sample,

            }
    evaluate.save(path_or_file="./results/", **result, **params)
    return result    
    


# %%
task = "Automatic Speech Recognition"
dataset = "CommonVoice-ES-10.0"
models = [
    '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_es',
    ]
metrics = ["wer", "cer"]

n_test = None

for model in models:
    print("===== Benchmark of model: {} -- dataset: {} ".format(model.split("/")[-1], dataset))
    asr_evaluate_metric_with_model_on_commonvoice(
        task=task,
        dataset=dataset,
        model=model,
        metrics=metrics,
        n_test=n_test
    )


