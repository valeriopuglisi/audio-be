import os.path
import yaml
Datasets = {
    "Automatic Speech Recognition": {
        "CommonVoice IT": {
            "name": "CommonVoice IT",
            "dataset_path": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it"),
            "test_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "test.tsv"),
            "train_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "train.tsv"),
            "other_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "other.tsv"),
            "reported_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "reported.tsv"),
            "validated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "validated.tsv"),
            "invalidated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-it", "it", "invalidated.tsv"),
            "description": 'Common Voice: A Massively-Multilingual Speech Corpu. \n'
                           'Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer,'
                           'Reuben Morais, Lindsay Saunders, Francis M. Tyers, Gregor Weber \n'
                           'The Common Voice corpus is a massively-multilingual collection of transcribed speech '
                           'intended for speech technology research and development. Common Voice is designed for '
                           'Automatic Speech Recognition purposes but can be useful in other domains '
                           '(e.g. language identification). To achieve scale and sustainability, '
                           'the Common Voice project employs crowdsourcing for both data collection and data '
                           'validation.  The most recent release includes 29 languages, and as of November 2019 there'
                           'are a total of  38 languages collecting data. Over 50,000 individuals have participated so '
                           'far, resulting in 2,500 hours of collected audio.To our knowledge this is the largest audio'
                           'corpus in the public domain for speech recognition, both in terms of number of hours and '
                           'number of  languages. As an example use case for Common Voice, we present speech '
                           'recognition experiments using Mozilla\'s DeepSpeech Speech-to-Text toolkit. '
                           'By applying transfer learning from a source English model, we find an average '
                           'Character Error Rate improvement of 5.99 +/- 5.48 for twelve target languages '
                           '(German, French, Italian, Turkish, Catalan, Slovenian, Welsh, Irish, Breton, Tatar, '
                           'Chuvash, and Kabyle). For most of these languages, these are the first ever published '
                           'results on end-to-end Automatic Speech Recognition.',
            "references": 'Ardila, Rosana, et al. "Common voice: A massively-multilingual speech corpus." '
                          'arXiv preprint arXiv:1912.06670 (2019).',
        },

        "CommonVoice EN": {
            "name": "CommonVoice EN",
            "path": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en"),
            "test_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "test.tsv"),
            "train_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "train.tsv"),
            "other_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "other.tsv"),
            "reported_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "reported.tsv"),
            "validated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "validated.tsv"),
            "invalidated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-en", "cv-corpus-9.0-2022-04-27", "en", "invalidated.tsv"),
            "description": 'Common Voice: A Massively-Multilingual Speech Corpu. \n'
                           'Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer,'
                           'Reuben Morais, Lindsay Saunders, Francis M. Tyers, Gregor Weber \n'
                           'The Common Voice corpus is a massively-multilingual collection of transcribed speech '
                           'intended for speech technology research and development. Common Voice is designed for '
                           'Automatic Speech Recognition purposes but can be useful in other domains '
                           '(e.g. language identification). To achieve scale and sustainability, '
                           'the Common Voice project employs crowdsourcing for both data collection and data '
                           'validation.  The most recent release includes 29 languages, and as of November 2019 there'
                           'are a total of  38 languages collecting data. Over 50,000 individuals have participated so '
                           'far, resulting in 2,500 hours of collected audio.To our knowledge this is the largest audio'
                           'corpus in the public domain for speech recognition, both in terms of number of hours and '
                           'number of  languages. As an example use case for Common Voice, we present speech '
                           'recognition experiments using Mozilla\'s DeepSpeech Speech-to-Text toolkit. '
                           'By applying transfer learning from a source English model, we find an average '
                           'Character Error Rate improvement of 5.99 +/- 5.48 for twelve target languages '
                           '(German, French, Italian, Turkish, Catalan, Slovenian, Welsh, Irish, Breton, Tatar, '
                           'Chuvash, and Kabyle). For most of these languages, these are the first ever published '
                           'results on end-to-end Automatic Speech Recognition.',
            "references": 'Ardila, Rosana, et al. "Common voice: A massively-multilingual speech corpus." '
                          'arXiv preprint arXiv:1912.06670 (2019).',
        },

        "CommonVoice DE":{
            "name": "CommonVoice DE",
            "path": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de"),
            "test_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "test.tsv"),
            "train_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "train.tsv"),
            "other_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "other.tsv"),
            "reported_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "reported.tsv"),
            "validated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "validated.tsv"),
            "invalidated_file": os.path.join("..", "datasets", "cv-corpus-9.0-2022-04-27-de", "cv-corpus-9.0-2022-04-27", "de", "invalidated.tsv"),
            "description": 'Common Voice: A Massively-Multilingual Speech Corpu. \n'
                           'Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer,'
                           'Reuben Morais, Lindsay Saunders, Francis M. Tyers, Gregor Weber \n'
                           'The Common Voice corpus is a massively-multilingual collection of transcribed speech '
                           'intended for speech technology research and development. Common Voice is designed for '
                           'Automatic Speech Recognition purposes but can be useful in other domains '
                           '(e.g. language identification). To achieve scale and sustainability, '
                           'the Common Voice project employs crowdsourcing for both data collection and data '
                           'validation.  The most recent release includes 29 languages, and as of November 2019 there'
                           'are a total of  38 languages collecting data. Over 50,000 individuals have participated so '
                           'far, resulting in 2,500 hours of collected audio.To our knowledge this is the largest audio'
                           'corpus in the public domain for speech recognition, both in terms of number of hours and '
                           'number of  languages. As an example use case for Common Voice, we present speech '
                           'recognition experiments using Mozilla\'s DeepSpeech Speech-to-Text toolkit. '
                           'By applying transfer learning from a source English model, we find an average '
                           'Character Error Rate improvement of 5.99 +/- 5.48 for twelve target languages '
                           '(German, French, Italian, Turkish, Catalan, Slovenian, Welsh, Irish, Breton, Tatar, '
                           'Chuvash, and Kabyle). For most of these languages, these are the first ever published '
                           'results on end-to-end Automatic Speech Recognition.',
            "references": 'Ardila, Rosana, et al. "Common voice: A massively-multilingual speech corpus." '
                          'arXiv preprint arXiv:1912.06670 (2019).',
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
# datasets_yaml = yaml.dump(Datasets)
# import pathlib
# datasets_references_path = "deep_learning_dict_datasets.yaml"
# with open(datasets_references_path, 'w+') as yamlfile:
#     yaml.safe_dump(datasets_yaml, yamlfile, default_flow_style=True)  # Also note the safe_dump
