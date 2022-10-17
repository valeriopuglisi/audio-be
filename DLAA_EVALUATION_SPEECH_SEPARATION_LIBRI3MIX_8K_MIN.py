# %%
from deep_learning_features_audio import *
from deep_learning_dict_api import AudioAnalysisAPI
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML
from torchmetrics import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio, SignalNoiseRatio, SignalDistortionRatio, PermutationInvariantTraining
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.functional.audio import signal_distortion_ratio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from datetime import datetime
from deep_learning_dict_datasets import Datasets
import numpy as np
import json


# %%
# When running this tutorial in Google Colab, install the required packages
# with the following.
# !pip install torchaudio librosa boto3

import torch
import torchaudio
import torchaudio.functional as TAF
import torchaudio.transforms as T

print(torch.__version__)
print(torchaudio.__version__)

# %%
def speech_separation_evaluate_metric_with_model_on_libri3mix(model, dataset, metrics, n_test, mix_type="test_mix_clean_file"):
    task = "Speech Separation"
    total_si_snr = torch.zeros(0)
    total_si_sdr = torch.zeros(0)
    total_snr = torch.zeros(0)
    total_sdr = torch.zeros(0)
    total_nb_pesq = torch.zeros(0)
    total_wb_pesq = torch.zeros(0)
    total_pit = torch.zeros(0)
    total_stoi= torch.zeros(0)
    metrics_error = 0 
    experiments = {
    }

    result = {}
    # os.walk(dataset_path)
    print("===================================================>  Dataset: {}".format(dataset))
    # print(Datasets[task])
    print("Datasets[task][dataset] : {}".format(Datasets[task][dataset][mix_type]))
    test_table = pd.read_table(Datasets[task][dataset][mix_type], sep=",")
    # display(test_table)
    # cols = test_table.iloc[:,1:4]
    print("len(test_table): {}".format(len(test_table)))
    # display(cols)
    for i, row in enumerate(test_table.iterrows()):
        preds = torch.zeros(0) 
        target = torch.zeros(0)
        print("====> Benchmarking: {}/{}   tot {}".format(i+1, n_test, test_table.shape[0]))
        # print(i, row[1]['mixture_path'], , , row[1]['noise_path'])

        model_channels = AudioAnalysisAPI[model]['channels']
        dataset_channels = Datasets[task][dataset]['channels']
        if model_channels != dataset_channels:
            return {"error": "Model and Dataset have differents channels number"}


        mixture_path = row[1]['mixture_path']
        source_1_path = row[1]['source_1_path']
        source_2_path = row[1]['source_2_path']
        source_3_path = row[1]['source_3_path']

        #Taking model sample rate
        model_sample_rate = AudioAnalysisAPI[model]['sample_rate']
        dataset_sample_rate = Datasets[task][dataset]['sample_rate']
        # print("model_sample_rate:{} - dataset_sample_rate:{}".format(model_sample_rate, dataset_sample_rate))
        # Writing on experiment original paths
        experiments[str(i)]= {
            "mixture_path":mixture_path,
            "source_1_path":source_1_path,
            "source_2_path":source_2_path,
            "source_3_path":source_3_path,
        }
        # Loading targets audio on tensors
        mixture_waveform, mixture_sample_rate = torchaudio.load(mixture_path)
        source_1_waveform, source_1_sample_rate = torchaudio.load(source_1_path)
        source_2_waveform, source_2_sample_rate = torchaudio.load(source_2_path)
        source_3_waveform, source_3_sample_rate = torchaudio.load(source_3_path)
        

        if model_sample_rate != dataset_sample_rate:
            mixture_waveform = TAF.resample(mixture_waveform, mixture_sample_rate, model_sample_rate)
            source_1_waveform = TAF.resample(source_1_waveform, source_1_sample_rate, model_sample_rate)
            source_2_waveform = TAF.resample(source_2_waveform, source_2_sample_rate, model_sample_rate)
            source_3_waveform = TAF.resample(source_3_waveform, source_3_sample_rate, model_sample_rate)
            
            mixture_path =  "libri3mix_8k_min_evaluation_target_mixture.wav"
            source_1_path = "libri3mix_8k_min_evaluation_target_source_1.wav"
            source_2_path = "libri3mix_8k_min_evaluation_target_source_2.wav"
            source_3_path = "libri3mix_8k_min_evaluation_target_source_3.wav"
            
            print("revisited mixture_path:{}".format(mixture_path))
            print("revisited source_1_path:{}".format(source_1_path))
            print("revisited source_2_path:{}".format(source_2_path))
            print("revisited source_3_path:{}".format(source_3_path))
            torchaudio.save(mixture_path, mixture_waveform, model_sample_rate)
            torchaudio.save(source_1_path, source_1_waveform, model_sample_rate)
            torchaudio.save(source_2_path, source_2_waveform, model_sample_rate)
            torchaudio.save(source_3_path, source_3_waveform, model_sample_rate)

            if mix_type == "test_mix_both_file":
                noise_path = row[1]['noise_path']
                noise_waveform, noise_sample_rate = torchaudio.load(noise_path)
                noise_waveform = TAF.resample(noise_waveform, noise_sample_rate, model_sample_rate)
                source_1_waveform = source_1_waveform + noise_waveform
                source_2_waveform = source_2_waveform + noise_waveform
                source_3_waveform = source_3_waveform + noise_waveform
        else:
            if mix_type == "test_mix_both_file":
                noise_path = row[1]['noise_path']
                noise_waveform, noise_sample_rate = torchaudio.load(noise_path)
                source_1_waveform = source_1_waveform + noise_waveform
                source_2_waveform = source_2_waveform + noise_waveform
                source_3_waveform = source_3_waveform + noise_waveform

        # Separate audio files with choosen model
        source_1_path_prediction, source_2_path_prediction, source_3_path_prediction = AudioAnalysisAPI[model]['function'](audiofile_path=mixture_path)
        
        # Loading predictions audio on tensors
        source_1_prediction_waveform, source_1_prediction_sample_rate = torchaudio.load(source_1_path_prediction)
        source_2_prediction_waveform, source_2_prediction_sample_rate = torchaudio.load(source_2_path_prediction)
        source_3_prediction_waveform, source_3_prediction_sample_rate = torchaudio.load(source_3_path_prediction)
        
            
        # Concatenating predictions into torch tensor 
        preds = torch.cat((preds, source_1_prediction_waveform), 0)
        preds = torch.cat((preds, source_2_prediction_waveform), 0)
        preds = torch.cat((preds, source_3_prediction_waveform), 0)
        # Concatenating targets into torch tensor 
        target = torch.cat((target, source_1_waveform), 0)
        target = torch.cat((target, source_2_waveform), 0)
        target = torch.cat((target, source_3_waveform), 0)
       
        # print("mixture_path : {}".format(mixture_path))
        # print("mixture_sample_rate:{}".format(mixture_sample_rate))
        # print("mixture_waveform.shape:{}".format(mixture_waveform.shape))
        # print("mixture_waveform:{}".format(mixture_waveform))
        # print()
        # print("source_1_path : {}".format(source_1_path))
        # print("source_1_sample_rate:{}".format(source_1_sample_rate))
        # print("source_1_waveform.shape:{}".format(source_1_waveform.shape))
        # print("source_1_waveform:{}".format(target[0]))
        # print()
        # print("source_1_path_prediction : {}".format(source_1_path_prediction))
        # print("source_1_prediction_sample_rate: {}".format(source_1_prediction_sample_rate))
        # print("source_1_prediction_waveform.shape : {}".format(source_1_prediction_waveform.shape))
        # print("source_1_prediction_waveform: {}".format(preds[0]))
        # print()
        # print("source_2_path : {}".format(source_2_path))
        # print("source_2_sample_rate:{}".format(source_2_sample_rate))
        # print("source_2_waveform.shape:{}".format(source_2_waveform.shape))
        # print("source_2_waveform:{}".format(target[1]))
        # print()
        # print("source_2_path_prediction : {}".format(source_2_path_prediction))
        # print("source_2_prediction_sample_rate: {}".format(source_2_prediction_sample_rate))
        # print("source_2_prediction_waveform.shape : {}".format(source_2_prediction_waveform.shape))
        # print("source_2_prediction_waveform: {}".format(preds[1]))
        # print()
        # print("source_3_path : {}".format(source_3_path))
        # print("source_3_sample_rate:{}".format(source_3_sample_rate))
        # print("source_3_waveform.shape:{}".format(source_3_waveform.shape))
        # print("source_3_waveform:{}".format(target[2]))
        # print()
        # print("source_3_path_prediction : {}".format(source_3_path_prediction))
        # print("source_3_prediction_sample_rate: {}".format(source_3_prediction_sample_rate))
        # print("source_3_prediction_waveform.shape : {}".format(source_3_prediction_waveform.shape))
        # print("source_3_prediction_waveform: {}".format(preds[2]))

        # print("preds.shape {}\ntarget.shape:{}".format(preds.shape, target.shape))
        # print("preds: {},\ntarget:{}".format(preds, target))
        
        experiments[str(i)]["source_1_path_prediction"] = source_1_path_prediction
        experiments[str(i)]["source_2_path_prediction"] = source_2_path_prediction
        experiments[str(i)]["source_3_path_prediction"] = source_3_path_prediction

        
        try:
            for metric in metrics:
            #print("Calculating metric:", metric)
                if metric == "si-snr":
                    si_snr = ScaleInvariantSignalNoiseRatio()
                    si_snr_result = si_snr(preds, target)
                    si_snr_result = torch.reshape(si_snr_result, (1, 1))
                    total_si_snr = torch.cat((total_si_snr, si_snr_result))
                    experiments[str(i)]["si-snr"] = str(si_snr_result)

                if metric == "si-sdr":
                    si_sdr = ScaleInvariantSignalDistortionRatio()
                    si_sdr_result = si_sdr(preds, target)
                    si_sdr_result = torch.reshape(si_sdr_result, (1, 1))
                    total_si_sdr = torch.cat((total_si_sdr, si_sdr_result)) 
                    experiments[str(i)]["si-sdr"] = str(si_sdr_result)                

                if metric == "snr":
                    snr = SignalNoiseRatio()
                    snr_result = snr(preds, target)
                    snr_result = torch.reshape(snr_result, (1, 1))
                    total_snr = torch.cat((total_snr, snr_result))
                    experiments[str(i)]["snr"] = str(snr_result)
                    
                if metric == "sdr":
                    sdr = SignalDistortionRatio()
                    sdr_result = sdr(preds, target)
                    sdr_result = torch.reshape(sdr_result, (1, 1))
                    total_sdr = torch.cat((total_sdr, sdr_result))
                    experiments[str(i)]["sdr"] = str(sdr_result)

                if metric == "pesq":
                    nb_pesq = PerceptualEvaluationSpeechQuality(model_sample_rate, 'nb')
                    nb_pesq_result = nb_pesq(preds, target)
                    nb_pesq_result = torch.reshape(nb_pesq_result, (1, 1))
                    total_nb_pesq = torch.cat((total_nb_pesq, nb_pesq_result))
                    experiments[str(i)]["pesq"] = str(nb_pesq_result)

                    if model_sample_rate > 8000: 
                        wb_pesq = PerceptualEvaluationSpeechQuality(model_sample_rate, 'wb')
                        wb_pesq_result = wb_pesq(preds, target)
                        wb_pesq_result = torch.reshape(wb_pesq_result, (1, 1))
                        total_wb_pesq = torch.cat((total_wb_pesq, wb_pesq_result))
                        experiments[str(i)]["wb-pesq"] = str(wb_pesq_result)

                if metric == "pit":
                    pit = PermutationInvariantTraining(signal_distortion_ratio, 'max')
                    pit_result = pit(preds, target)
                    pit_result = torch.reshape(pit_result, (1, 1))
                    total_pit = torch.cat((total_pit, pit_result))
                    experiments[str(i)]["pit"] = str(pit_result)

                if metric == "stoi":
                    stoi_src = ShortTimeObjectiveIntelligibility(model_sample_rate, False)
                    stoi_result = stoi_src(preds, target)
                    stoi_result = torch.reshape(stoi_result, (1, 1))
                    total_stoi = torch.cat((total_stoi, stoi_result))
                    experiments[str(i)]["stoi"] = str(stoi_result)
        except Exception as e:
            print("=====> ERROR: {}".format(e))
            metrics_error += 1



        if i == n_test - 1: 
            break

    
    total_si_snr = torch.sum(total_si_snr)/n_test
    total_si_sdr = torch.sum(total_si_sdr)/n_test
    total_snr = torch.sum(total_snr)/n_test
    total_sdr = torch.sum(total_sdr)/n_test
    total_pit = torch.sum(total_pit)/n_test
    total_wb_pesq = torch.sum(total_wb_pesq)/n_test
    total_nb_pesq = torch.sum(total_nb_pesq)/n_test
    total_stoi = torch.sum(total_stoi)/n_test

    print("============================================================================================")
    print("total_si_snr:{}".format(total_si_snr)) 
    print("total_si_sdr:{}".format(total_si_sdr)) 
    print("total_snr:{}".format(total_snr)) 
    print("total_sdr:{}".format(total_sdr)) 
    print("total_pit:{}".format(total_pit)) 
    print("total_wb_pesq:{}".format(total_wb_pesq))  
    print("total_nb_pesq:{}".format(total_nb_pesq))  
    print("total_stoi:{}".format(total_stoi))  
    # Creating the content of result file
    result ={
        "model": model, 
        "dataset": dataset, 
        "n_test": n_test,
        "metrics_error": metrics_error,
        "model_sample_rate":model_sample_rate,
        "dataset_sample_rate":dataset_sample_rate,
        "n_test_done": n_test - metrics_error,
        "total_si_snr":str(total_si_snr),
        "total_si_sdr":str(total_si_sdr),
        "total_snr":str(total_snr),
        "total_sdr":str(total_sdr),
        "total_pit":str(total_pit),
        "total_wb_pesq":str(total_wb_pesq),
        "total_nb_pesq":str(total_nb_pesq),
        "total_stoi":str(total_stoi),
        "experiments": experiments,
    }

    # Creating filename
    dateTimeObj = datetime.now()
    print(dateTimeObj)
    timestampStr = dateTimeObj.strftime("%d_%b_%Y__%H_%M_%S_%f")
    result_filename = timestampStr + "_evaluate_" + model.split("/")[-1] + "_" + dataset + "_" + mix_type + "_" + str(n_test)+".json"
    print('result_filename : ', result_filename)
    with open(result_filename, 'w') as f:
        json.dump(result, f)
        
    return result, n_test

speech_separation_3_channels_models = ['/api/audioseparation/speech_separation_sepformer_wsj03mix']
speech_separation_dataset = ["Libri3Mix8kMin", "Libri3Mix8kMax", "Libri3Mix16kMin", "Libri3Mix16kMax"]
metrics = ["si-snr", "si-sdr", "sdr", "snr", "pesq", "stoi"]
n_test = 3000

# type = "test_mix_clean_file" #"test_mix_both_file" "test_mix_single_file"
type = "test_mix_both_file" 
speech_separation_evaluate_metric_with_model_on_libri3mix(speech_separation_3_channels_models[0], speech_separation_dataset[0], metrics, n_test=n_test, mix_type=type )



