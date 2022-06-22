from deep_learning_audio_features import *
from deep_learning_dict_api import AudioAnalysisAPI
from deep_learning_dict_datasets import Datasets
import evaluate
from evaluate import load
import pandas as pd
from pathlib import Path


def asr_evaluate_metric_on_dataset(task, dataset, model, metrics):
    predictions = []
    references = []
    result = {}
    # os.walk(dataset_path)
    test_table = pd.read_table(Datasets[task][dataset]["test_file"])
    test_audio_path = Datasets[task][dataset]["dataset_path"]
    for i, row in enumerate(test_table.iterrows()):
        print("Benchmarking: {}/{}".format(i, test_table.shape[0]))
        audiofile_path = row[1]["path"]
        wav_audiofile_path = os.path.splitext(audiofile_path)[0] + '.wav'
        reference = row[1]["sentence"].lower()
        audio_path = os.path.join(test_audio_path, "wavs", wav_audiofile_path)
        prediction = AudioAnalysisAPI[model]['function'](audiofile_path=audio_path)
        predictions.append(prediction)
        references.append(reference)
        # print("audiofile_path: {}\n- reference: {}\n- prediction:{}\n".format(audio_path, reference, prediction))
    for metric in metrics:
        loaded_metric = load(metric)
        # wer = load("wer")
        caluculated_metric = loaded_metric.compute(predictions=predictions, references=references)
        # wer_score = wer.compute(predictions=predictions, references=references)
        print("{}: {}".format(metric, caluculated_metric))
        # print("wer_score: {}".format(wer_score))
        result[metric] = caluculated_metric
    params = {"model": model}
    evaluate.save(path_or_file="./results/", **result, **params)
    return result


task = "Automatic Speech Recognition"
dataset = "CommonVoice IT"
models = [
    '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_it',
    '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_it',
    '/api/automatic_speech_recognition/asr__crdnn__commonvoice_it']
metrics = ["wer", "cer"]


asr_evaluate_metric_on_dataset(
        task=task,
        dataset=dataset,
        model=models[0],
        metrics=metrics,
    )
# for model in models:
#     print("===== Benchmark of model: {} dataset: {} ".format(model.split("/")[-1], dataset))
#     asr_evaluate_metric_on_dataset(
#         task=task,
#         dataset=dataset,
#         model=model,
#         metrics=metrics,
#     )
