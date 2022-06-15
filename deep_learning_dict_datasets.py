import os.path
import yaml
Datasets = {
    "Automatic Speech Recognition": {
        "CommonVoice IT": {
            "dataset_path": os.path.join("..", "datasets", "COMMON VOICE IT", "it"),
            "test_file": os.path.join("..", "datasets", "COMMON VOICE IT", "it", "test.tsv"),
            "description": "a",
            "references": "b",
        },

        "CommonVoice EN": {
            "path": os.path.join("..", "datasets", "COMMON VOICE EN"),
            "description": "a",
            "references": "b",
        },

        "CommonVoice DE":{
            "path": os.path.join("..", "datasets", "COMMON VOICE DE"),
            "description": "a",
            "references": "b",
        }
    },
    "Emotion Recognition":{

    },
    "Language Identification":{

    },
    "Voice Activity Detection":{

    },
    "Speech Enhancement":{

    },
    "Speech Separation": {

    }
}
datasets_yaml = yaml.dump(Datasets)
import pathlib
datasets_references_path = "deep_learning_dict_datasets.yaml"
with open(datasets_references_path, 'w+') as yamlfile:
    yaml.safe_dump(datasets_yaml, yamlfile, default_flow_style=True)  # Also note the safe_dump
