import json
from pprint import pprint

from deep_learning_dict_api import AudioAnalysisAPI

api_list = AudioAnalysisAPI
for dict_element in api_list.values():
    print(type(dict_element))
    print(dict_element)
    del dict_element['function']
    print(dict_element)
    print()
dl_api_list = json.dumps(api_list)
pprint(dl_api_list)