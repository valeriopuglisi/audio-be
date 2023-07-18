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
def language_identification_evaluate_metric_with_model_on_voxlingua_107(
    task, 
    dataset, 
    model, 
    metrics, 
    n_test,
    languages_to_test):
    tot_sample = 0 
    test_done = 0
    errors = 0
    predictions = []
    references = []
    result = {
        "evaluation": {}
    }

    languages_dict ={}

    ecapa_languages_dict = {
        "Arabic":"ar",
        "Basque":1,
        "Breton":2,
        "Catalan":3,
        "Chinese_China":4,
        "Chinese_Hongkong":5,
        "Chinese_Taiwan":6,
        "Chuvash":7,
        "Czech":8,
        "Dhivehi":9,
        "Dutch":10,
        "English":"en",
        "Esperanto":12,
        "Estonian":13,
        "French":"fr",
        "Frisian":15,
        "Georgian":16,
        "German":"de",
        "Greek":18,
        "Hakha_Chin":19,
        "Indonesian":20,
        "Interlingua":21,
        "Italian":"it",
        "Japanese":"ja",
        "Kabyle":24,
        "Kinyarwanda":25,
        "Kyrgyz":26,
        "Latvian":"lv",
        "Maltese":28,
        "Mangolian":29,
        "Persian":30,
        "Polish":31,
        "Portuguese":32,
        "Romanian":33,
        "Romansh_Sursilvan":34,
        "Russian":"ru",
        "Sakha":35,
        "Slovenian":37,
        "Spanish":"es",
        "Swedish":39,
        "Tamil":40,
        "Tatar":41,
        "Turkish":42,
        "Ukranian":43,
        "Welsh":44,
    }


    # os.walk(dataset_path)
    datset_path = Datasets[task][dataset]["path"]
    
    for i, lang_dir_path in enumerate(os.listdir(datset_path)):
        dataset_lang_dir_path = os.path.join(datset_path, lang_dir_path)
        languages_dict[lang_dir_path] = i 


    for i, lang_dir_path in enumerate(languages_to_test):
        
        tot_sample +=1
        dataset_lang_dir_path = os.path.join(datset_path, lang_dir_path)
        print("languages_dict: ", languages_dict)
        print("lang_dir_path:{}".format(lang_dir_path))
        for i, filename in enumerate(os.listdir(dataset_lang_dir_path)):
            
            audio_path = os.path.join(dataset_lang_dir_path, filename)
            print("==> audio_path:{}".format(audio_path))
            print("==> filename:{}".format(filename))
            print("Benchmarking: {}/{}".format(i, len(os.listdir(dataset_lang_dir_path))))
            
            if i == n_test:
                    break
           
            if os.path.isfile(audio_path) and lang_dir_path in languages_dict:
                
                prediction = AudioAnalysisAPI[model]['function'](audiofile_path=audio_path)
                
                if prediction is not None and prediction != "":
                    print("==> prediction :", prediction[0])
                    try:
                        language_index_reference = languages_dict[lang_dir_path]
                        
                        if model == '/api/language_id/langid_voxlingua107_ecapa':
                            prediction[0] = prediction[0][4:]

                        language_index_prediction = languages_dict[ecapa_languages_dict[prediction[0]]]
                    except:
                        language_index_reference = languages_dict[lang_dir_path]
                        language_index_prediction = language_index_reference + 1

                    predictions.append(language_index_prediction)
                    references.append(language_index_reference)
                    print("audiofile_path: {}".format(audio_path))
                    print("reference: {}, index:{}".format(lang_dir_path, language_index_reference))
                    print(" ecapa_languages_dict_prediction:{}, prediction:{}, index:{}".format(ecapa_languages_dict[prediction[0]], prediction[0], language_index_prediction))
                    print()
                    test_done += 1
            else:
                print(audio_path, " file doesn't exist")
                errors+=1
           
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
        # "n_test": n_test,
        # "n_test_done": test_done,
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
task = "Language Identification"
dataset = "VoxLingua107"
models = [
    '/api/language_id/langid_commonlanguage_ecapa',
    '/api/language_id/langid_voxlingua107_ecapa'
    ]
metrics = ["accuracy"]
languages_to_test =["de", "en", "es", "fr", "it",]
n_test = None

for model in models:
    print("===== Benchmark of model: {} -- dataset: {} ".format(model.split("/")[-1], dataset))
    language_identification_evaluate_metric_with_model_on_voxlingua_107(
        task=task,
        dataset=dataset,
        model=model,
        metrics=metrics,
        n_test=n_test,
        languages_to_test=languages_to_test
    )


