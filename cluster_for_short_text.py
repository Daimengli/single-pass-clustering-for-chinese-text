#! /usr/bin/python

# -*- coding: utf-8 -*-

import numpy as np
import argparse
import json
from fuzzywuzzy import fuzz
from collections import defaultdict
from tqdm import tqdm
import re

def load_corpus(filepath, sep="\t"):
    corpus = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(sep)[0]
            corpus.append(line)
    return corpus


def write_cluster(cluster_topic, filepath):
    with open(filepath, "w") as w:
        for key, value in cluster_topic.items():
            cluster = [(text, sim) for text, sim in value if text != key]
            cluster = json.dumps(cluster, ensure_ascii=False)
            w.write("{}\t{}\n".format(key, cluster))

#def fuzz_sim(text1, text2, lower=True):
#    if lower:
#        text1, text2 = text1.lower(), text2.lower()
#    partial_ratio = fuzz.partial_ratio(text1, text2)/100
#    simple_ratio = fuzz.ratio(text1, text2)/100
#    return 0.8*partial_ratio + 0.2*simple_ratio

def hit_head(text1, text2):
    shorter = text1 if len(text1) < len(text2) else text2
    longer = text1 if len(text1) > len(text2) else text2
    pattern = "^" + shorter
    return any(re.findall(pattern, longer))

def adjust_score(partial_ratio, text1, text2):
    min_len = min(len(text1), len(text2))
    if partial_ratio >= 0.8 and min_len >= 4:
        partial_ratio += 0.15
    if partial_ratio >= 0.8 and min_len <= 2:
        partial_ratio -= 0.15
        head_hit = hit_head(text1, text2)
        if head_hit:
            partial_ratio += 0.15
        else:
            partial_ratio -= 0.15
    if min_len == 1:
        partial_ratio = 0.0
    return partial_ratio
    
def fuzz_sim(text1, text2, lower=True):
    if lower:
        text1, text2 = text1.lower(), text2.lower()
    partial_ratio = fuzz.partial_ratio(text1, text2)/100
    partial_ratio = adjust_score(partial_ratio, text1, text2)
    simple_ratio = fuzz.ratio(text1, text2)/100
    return min(1.0, 0.8*partial_ratio + 0.2*simple_ratio)



def get_max_similarity(cluster_topic, text, sim_func):
    max_value = 0
    max_index = -1
    for k, cluster in cluster_topic.items():
#        similarity = np.mean([sim_func(text, v) for v, s in cluster])
        similarity = sim_func(text, k)
        if similarity > max_value:
            max_value = similarity
            max_index = k
    return max_index, max_value


def single_pass(corpus, sim_func, thres, sort_by_length=True):
    if isinstance(corpus, str):
        corpus = load_corpus(corpus)
        
    if sort_by_length:
        corpus = sorted(corpus, key=lambda x:len(x))

    cluster_topic = defaultdict(list)
    for text in tqdm(corpus): 
        if len(cluster_topic) == 0:
            cluster_topic[text].append((text, 1.0))  
        else:
            max_index, max_value = get_max_similarity(cluster_topic, text, 
                                                      sim_func)
#            print(text, max_index, max_value)
            #join the most similar topic
            if max_value >= thres:
                cluster_topic[max_index].append((text, max_value))
            #else create the new topic
            else:
                cluster_topic[text].append((text, 1.0))
                
    return cluster_topic

def main():
    parser = argparse.ArgumentParser()
    
    ### Required parameters
    parser.add_argument("--corpus_file", default=None, type=str, required=True,
                        help="corpus file, one line one short text.")
    parser.add_argument("--sim_func", default=None, type=str, required=True,
                        help="similarity function")
    parser.add_argument('--threshold', type=float, default=0.0, required=True,
                        help="similarity threshold.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file.")
    args = parser.parse_args()
    
    sim_func_dict = {"fuzz":fuzz_sim}
    sim_func = sim_func_dict[args.sim_func]
    
    cluster_topic = single_pass(args.corpus_file, sim_func, args.threshold)
    write_cluster(cluster_topic, args.output_file)

if __name__ == '__main__':
    main()
