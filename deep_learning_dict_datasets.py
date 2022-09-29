import os.path
import yaml
Datasets = {
    "Automatic Speech Recognition": {  
        "CommonVoice-DE-9.0":{
            "name": "CommonVoice-DE-9.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/de/invalidated.tsv",
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
        "CommonVoice-EN-9.0": {
            "name": "CommonVoice-EN-9.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/en/invalidated.tsv",
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
        "CommonVoice-ES-9.0": {
            "name": "CommonVoice-ES-9.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/es/invalidated.tsv",
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
        "CommonVoice-FR-9.0": {
            "name": "CommonVoice-FR-9.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/fr/invalidated.tsv",
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
        "CommonVoice-IT-9.0": {
            "name": "CommonVoice-IT-9.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it",
            "test_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/test.tsv",
            "train_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/train.tsv",
            "other_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/other.tsv",
            "reported_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/reported.tsv",
            "validated_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/validated.tsv",
            "invalidated_file": "/storage/data_8T/datasets/audio/common-voice-corpus-9.0-2022-04-27/it/invalidated.tsv",
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
        "CommonVoice-DE-10.0":{
            "name": "CommonVoice-DE-10.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/de/invalidated.tsv",
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
        "CommonVoice-EN-10.0": {
            "name": "CommonVoice-EN-10.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/en/invalidated.tsv",
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
        "CommonVoice-ES-10.0": {
            "name": "CommonVoice-ES-10.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/es/invalidated.tsv",
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
        "CommonVoice-FR-10.0": {
            "name": "CommonVoice-FR-10.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr",
            "test_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/other.tsv",
            "reported_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/reported.tsv",
            "validated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/validated.tsv",
            "invalidated_file":"/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/fr/invalidated.tsv",
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
        "CommonVoice-IT-10.0": {
            "name": "CommonVoice-IT-10.0",
            "path": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it",
            "test_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/test.tsv",
            "train_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/train.tsv",
            "other_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/other.tsv",
            "reported_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/reported.tsv",
            "validated_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/validated.tsv",
            "invalidated_file": "/storage/data_8T/datasets/audio/common-voice-corpus-10.0-2022-07-04/it/invalidated.tsv",
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
        "LibriSpeech": {
            "name": "LibriSpeech",
            "path": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech",
            "dev_clean": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/dev-clean",
            "dev_other": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/dev-other",
            "test_clean": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/test-clean",
            "test_other": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/test-other",
            "train_clean_100": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/train-clean-100",
            "train_clean_360": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/train-clean-360",
            "train_other_500": "/storage/data_8T/datasets/audio/LibriMix/LibriSpeech/train_other_500",
            "description": 
                'This paper introduces a new corpus of read English speech, suitable for training and evaluating'
                'speech recognition systems. The LibriSpeech corpus is derived from audiobooks that are part of the LibriVox' 
                'project, and contains 1000 hours of speech sampled at 16 kHz. We have made the corpus freely available for download,'
                'along with separately prepared language-model training data and pre-built language models. '
                'We show that acoustic models trained on LibriSpeech give lower error rate on the Wall Street Journal (WSJ)'
                'test sets than models trained on WSJ itself. '
                'We are also releasing Kaldi scripts that make it easy to build these systems.',
            "references": 'Panayotov, Vassil, et al. "Librispeech: an asr corpus based on public domain audio books." 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015.',
        },

        "Voxpopuli-DE":{
            "name": "Voxpopuli-DE", 
            "path": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/de",
            "test_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/de/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/de/train.tsv",
            "dev_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/de/other.tsv",
            "description": 'We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages. It is the largest open data to date for unsupervised representation learning as well as semi-supervised learning. VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours. We provide speech recognition (ASR) baselines and validate the versatility of VoxPopuli unlabeled data in semi-supervised ASR and speech-to-text translation under challenging out-of-domain settings. The corpus is available at https://github.com/facebookresearch/voxpopuli.',
            "references": 'Wang, Changhan, et al. "Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation." arXiv preprint arXiv:2101.00390 (2021).',
        },
        "Voxpopuli-EN": {
            "name": "Voxpopuli-EN",
            "path": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/en",
            "test_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/en/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/en/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/en/other.tsv",
            "description": 'We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages. It is the largest open data to date for unsupervised representation learning as well as semi-supervised learning. VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours. We provide speech recognition (ASR) baselines and validate the versatility of VoxPopuli unlabeled data in semi-supervised ASR and speech-to-text translation under challenging out-of-domain settings. The corpus is available at https://github.com/facebookresearch/voxpopuli.',
            "references": 'Wang, Changhan, et al. "Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation." arXiv preprint arXiv:2101.00390 (2021).',
        },
        "Voxpopuli-ES": {
            "name": "Voxpopuli-ES",
            "path": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/es",
            "test_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/es/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/es/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/es/other.tsv",
            "description": 'We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages. It is the largest open data to date for unsupervised representation learning as well as semi-supervised learning. VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours. We provide speech recognition (ASR) baselines and validate the versatility of VoxPopuli unlabeled data in semi-supervised ASR and speech-to-text translation under challenging out-of-domain settings. The corpus is available at https://github.com/facebookresearch/voxpopuli.',
            "references": 'Wang, Changhan, et al. "Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation." arXiv preprint arXiv:2101.00390 (2021).',
        },
        "Voxpopuli-FR": {
            "name": "Voxpopuli-FR",
            "path": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/fr",
            "test_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/fr/test.tsv",
            "train_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/fr/train.tsv",
            "other_file":"/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/fr/other.tsv",
            "description": 'We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages. It is the largest open data to date for unsupervised representation learning as well as semi-supervised learning. VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours. We provide speech recognition (ASR) baselines and validate the versatility of VoxPopuli unlabeled data in semi-supervised ASR and speech-to-text translation under challenging out-of-domain settings. The corpus is available at https://github.com/facebookresearch/voxpopuli.',
            "references": 'Wang, Changhan, et al. "Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation." arXiv preprint arXiv:2101.00390 (2021).',
        },

        "Voxpopuli-IT": {
            "name": "Voxpopuli-IT",
            "path": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/it",
            "test_file": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/it/test.tsv",
            "train_file": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/it/train.tsv",
            "other_file": "/storage/data_8T/datasets/audio/voxpopuli/transcribed_data/it/other.tsv",
            "description": 'We introduce VoxPopuli, a large-scale multilingual corpus providing 400K hours of unlabeled speech data in 23 languages. It is the largest open data to date for unsupervised representation learning as well as semi-supervised learning. VoxPopuli also contains 1.8K hours of transcribed speeches in 15 languages and their aligned oral interpretations into 15 target languages totaling 17.3K hours. We provide speech recognition (ASR) baselines and validate the versatility of VoxPopuli unlabeled data in semi-supervised ASR and speech-to-text translation under challenging out-of-domain settings. The corpus is available at https://github.com/facebookresearch/voxpopuli.',
            "references": 'Wang, Changhan, et al. "Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation." arXiv preprint arXiv:2101.00390 (2021).',
        },

        
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
        "Libri2Mix8kMin": {
            "name": "Libri2Mix8kMin",
            "channels":2,
            "sample_rate": 8000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/min/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/min/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/min/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri2Mix8kMax": {
            "name": "Libri2Mix8kMax",
            "channels":2,
            "sample_rate": 8000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/max/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/max/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav8k/max/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri2Mix16kMin": {
            "name": "Libri2Mix16kMin",
            "channels":2,
            "sample_rate": 16000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/min/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri2Mix16kMax": {
            "name": "Libri2Mix16kMax",
            "channels":2,
            "sample_rate": 16000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri3Mix8kMin": {
            "name": "Libri3Mix8kMin",
            "channels":3,
            "sample_rate": 8000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/min/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/min/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/min/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri3Mix8kMax": {
            "name": "Libri3Mix8kMax",
            "channels":3,
            "sample_rate": 8000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/max/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/max/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav8k/max/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri3Mix16kMin": {
            "name": "Libri3Mix16kMin",
            "channels":3,
            "sample_rate": 16000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/min/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/min/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/min/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        "Libri3Mix16kMax": {
            "name": "Libri3Mix16kMax",
            "channels":3,
            "sample_rate": 16000,
            "path": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k",
            "test_mix_both_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/max/metadata/mixture_test_mix_both.csv",
            "test_mix_clean_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/max/metadata/mixture_test_mix_clean.csv",
            "test_mix_single_file": "/storage/data_8T/datasets/audio/LibriMix/Libri3Mix/wav16k/max/metadata/mixture_test_mix_single.csv",
            "description": 'In recent years, wsj0-2mix has become the reference dataset for single-channel speech separation. Most deep learning-based speech separation models today are benchmarked on it.' 
            'However, recent studies have shown important performance drops when models trained on wsj0-2mix are evaluated on other, similar datasets. To address this generalization issue, we created LibriMix, an open-source alternative to wsj0-2mix, and to its noisy extension, WHAM!.' 
            'Based on LibriSpeech, LibriMix consists of two- or three-speaker mixtures combined with ambient noise samples from WHAM!' 
            'In order to fairly evaluate across datasets, we introduce a third test set based on VCTK for speech and WHAM! for noise.' 
            'Our experiments show that the generalization error is smaller for models trained with LibriMix than with WHAM!, in both clean and noisy conditions. Aiming towards evaluation in more realistic,' 
            'conversation-like scenarios, we also release a sparsely overlapping version of LibriMix\'s test set.',
            "references": 'Cosentino, Joris, et al. "Librimix: An open-source dataset for generalizable speech separation." arXiv preprint arXiv:2005.11262 (2020).',
        },
        
        
       
       
        
        
        

    }
}
# datasets_yaml = yaml.dump(Datasets)
# import pathlib
# datasets_references_path = "deep_learning_dict_datasets.yaml"
# with open(datasets_references_path, 'w+') as yamlfile:
#     yaml.safe_dump(datasets_yaml, yamlfile, default_flow_style=True)  # Also note the safe_dump
