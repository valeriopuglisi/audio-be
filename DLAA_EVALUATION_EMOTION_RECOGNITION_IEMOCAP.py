# %%
from deep_learning_features_audio import *
from deep_learning_dict_api import AudioAnalysisAPI
from deep_learning_dict_datasets import Datasets
import evaluate
from evaluate import load
import pandas as pd
from pathlib import Path
import torch
from torchmetrics import Accuracy
from datetime import datetime
import json


# %%
def emotion_recognition_evaluate_metric_with_model_on_IEMOCAP(task, dataset, model, metrics, n_test):
    test_done = 0
    errors = 0
    predictions = []
    references = []
    result = {
        "evaluation": {}
    }

    emotions_dict ={
        "ang":0,
        "dis":1, 
        "exc":2, 
        "fea":3,
        "fru":4,
        "hap":5,
        "neu":6,
        "oth":7,	
        "sad":8,
        "sur":9,
        "xxx":10,
    }
    # os.walk(dataset_path)
    test_table = pd.read_csv(Datasets[task][dataset]["path"])
    tot_sample = test_table.shape[0]
    if n_test is None:
        n_test = tot_sample
    for i, row in enumerate(test_table.iterrows()):
        print("Benchmarking: {}/{}".format(i, test_table.shape[0]))
        # print("row 0 :{}".format(row[0]))
        # print("row 1 :{}".format(row[1]["audio_path"]))

        audio_path = row[1]["audio_path"]
        reference = row[1]["emotion_evaluation_1"]
        # print("audio_path:{}".format(audio_path))
        # print("reference:{}".format(reference))
        if i == n_test:
                break
        try:
            if os.path.isfile(audio_path):
                prediction = AudioAnalysisAPI[model]['function'](audiofile_path=audio_path)
                if prediction is not None and prediction != "":
                    emotion_index_prediction = emotions_dict[prediction[0]]
                    emotion_index_reference = emotions_dict[reference]
                    predictions.append(emotion_index_prediction)
                    references.append(emotion_index_reference)
                    print("audiofile_path: {}".format(audio_path))
                    print("reference: {}, index:{}".format(reference, emotion_index_reference))
                    print("prediction:{}, index:{}".format(prediction[0], emotion_index_prediction))
                    print()
                    test_done += 1
            else:
                print(audio_path, " file doesn't exist")
                errors+=1
            
        except Exception as e:
            print(e)
            errors += 1
            pass

    target = torch.tensor(references)
    preds = torch.tensor(predictions)
    accuracy = Accuracy()
    accuracy_result = accuracy(preds, target)
    print("==> accuracy_result:{}".format(accuracy_result))
    # Creating the content of result file
    result ={
        "model": model, 
        "dataset": dataset, 
        "tot_sample":tot_sample,
        "n_test": n_test,
        "n_test_done": test_done,
        "accuracy":str(accuracy_result),
    }

    # Creating filename
    dateTimeObj = datetime.now()
    print(dateTimeObj)
    timestampStr = dateTimeObj.strftime("%d_%b_%Y__%H_%M_%S_%f")
    result_filename = timestampStr + "_evaluate_" + model.split("/")[-1] + "_" + dataset + "_" + str(n_test)+".json"
    print('result_filename : ', result_filename)
    with open(result_filename, 'w') as f:
        json.dump(result, f)
    
    return result    
    


# %%
task = "Emotion Recognition"
dataset = "IEMOCAP"
models = [
    '/api/emotion_recognition/wav2vec2_IEMOCAP',
    ]
metrics = ["accuracy"]

n_test = None

for model in models:
    print("===== Benchmark of model: {} -- dataset: {} ".format(model.split("/")[-1], dataset))
    emotion_recognition_evaluate_metric_with_model_on_IEMOCAP(
        task=task,
        dataset=dataset,
        model=model,
        metrics=metrics,
        n_test=n_test
    )


