import json
import os
import sys
from tqdm import tqdm

def read_jsonl(file_path, key = None, num = None):
    if key is None:
        data = []
    else:
        data = {}

    with open(file_path, 'r') as file:
        if num is None:
            loader = tqdm(file)
        else:
            loader = tqdm(file, total = num)

        for line in loader:
            if key is None:
                data.append(json.loads(line))
            else:
                json_data = json.loads(line)
                data[json_data[key]] = json_data

    return data

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__