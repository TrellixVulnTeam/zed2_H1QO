import numpy as np
import pandas as pd
import json
import codecs
from collections import OrderedDict
def create_dt(fn):
    data_dict = []
    label_lst, sentence_lst = [], []
    df = pd.read_csv(fn, sep=',', header=None,encoding='utf-8')
    for dt in df.values.tolist():
        dicti={}
        label = dt[0]
        sentences = dt[1:]
        # label_lst.append(label)
        # sentence_lst.append(sentences)
        dicti['guidance'] = label
        sentences = [s.replace('相談者:', '') for s in sentences]
        sentences = [s.replace('相談員:', '') for s in sentences]
        dicti['sentences'] = sentences
        data_dict.append(dicti)
    return data_dict
def write_json(path,data_dict):
    hoge = codecs.open(path, 'w', 'shift_jis')
    json.dump(data_dict, hoge, ensure_ascii=False, indent=3)
    # json.dump(data_dict, hoge, indent=3)
#
def read_json(path):
    fd = open(path, mode='r')
    data = json.load(fd)
    fd.close()
    return data

if __name__ == "__main__":
    fn='C:/00_work/05_src/nlg/099_doc/remarktojson/remark_3.txt'
    data_dict=create_dt(fn)
    fn_json='C:/00_work/05_src/nlg/099_doc/remarktojson/remark_02.json'
    write_json(fn_json,data_dict)

    data=read_json(fn_json)
    for k in data:
        print(k['guidance'])
    # k=0