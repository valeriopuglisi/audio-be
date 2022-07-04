import os.path
import yaml
Metrics = {
    "Automatic Speech Recognition": {
        "WER": {
            "name": "WER: World Error Rate",
            "description": """Word error rate (WER) is a common metric of the performance of an
                automatic speech recognition system. The general difficulty of measuring per-
                formance lies in the fact that the recognized word sequence can have a different
                length from the reference word sequence (supposedly the correct one). The
                WER is derived from the Levenshtein distance , working at the
                word level instead of the phoneme level. The WER is a valuable tool for com-
                paring different systems as well as for evaluating improvements within one
                system. This kind of measurement, however, provides no details on the nature
                of translation errors and further work is therefore required to identify the main
                source(s) of error and to focus any research effort. This problem is solved by
                first aligning the recognized word sequence with the reference (spoken) word
                sequence using dynamic string alignment. Examination of this issue is seen
                through a theory called the power law that states the correlation between
                perplexity and word error rate. Word error rate can then be computed as:
                WER = (S + D + I)/N = (S + D + I)/(S + D + C) (1)
                where S is the number of substitutions, D is the number of deletions, I is the
                number of insertions, C is the number of correct words, N is the number of
                words in the reference (N = S + D + C).
                This value indicates the average number of errors per reference word. The
                lower the value, the better the performance of the ASR system with a WER
                of 0 being a perfect score.""",
        },
        "CER": {
                    "name": "WER: World Error Rate",
                    "description": """Character error rate (CER) is a common metric of the performance of an auto-
                    matic speech recognition system. CER is similar to Word Error Rate (WER),
                    but operates on character instead of word. Please refer to docs of WER for
                    further information. Character error rate can be computed as:
                    CER = (S + D + I)/N = (S + D + I)/(S + D + C) (2)
                    where:
                    - S is the number of substitutions,
                    - D is the number of deletions,
                    - I is the number of insertions,
                    - C is the number of correct characters,
                    - N is the number of characters in the reference (N = S + D + C).
                    CER’s output is not always a number between 0 and 1, in particular when
                    there is a high number of insertions. This value is often associated to the
                    """,
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

    }
}
# datasets_yaml = yaml.dump(Datasets)
# import pathlib
# datasets_references_path = "deep_learning_dict_datasets.yaml"
# with open(datasets_references_path, 'w+') as yamlfile:
#     yaml.safe_dump(datasets_yaml, yamlfile, default_flow_style=True)  # Also note the safe_dump
