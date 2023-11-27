import re
import os
from typing import List
from nltk.tokenize import word_tokenize


def tokenize_nltk(text):
    words = word_tokenize(text)
    output_list = []
    for w in words:
        w_list = re.findall(r'\w+', w)
        output_list.extend(w_list)
    return output_list


def file_distance(src_file, dest_file):
    distance = -1
    try:
        commonpath = os.path.commonpath([src_file, dest_file])
        rel_file1_path = os.path.relpath(src_file, commonpath)
        rel_file2_path = os.path.relpath(dest_file, commonpath)
        distance = rel_file1_path.count(os.sep) + rel_file2_path.count(os.sep)
    except Exception as e:
        # print(e, src_file, dest_file)
        pass

    return distance
