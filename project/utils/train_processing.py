#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from model.train import Train
from model.score import Score
from model.header import Header

import pickle

def save_training_to_mongo(train, label, thuat_toan, khoa):
    try:
        labels = label.tolist()
        for x, _ in enumerate(train):
            label_value = labels[x]
            
            data = {
                "position": x,
                "label": label_value,
                "identify": khoa + "_" + thuat_toan
            }
            label = Train.find_one(query={
                    "position": x,
                    "identify": khoa + "_" + thuat_toan
                })
            if label == -1:
                Train.insert(data=data)
            else:
                Train.update_one(query={"position": x, "identify": khoa + "_" + thuat_toan},
                                change={"label": label_value})
        return "okke"
    except Exception as e:
        print("err save_training_to_mongo: ", e)
        return "0"

def save_header(headers, thuat_toan, khoa):
    try:
        data = {
            "header": headers,
            "identify": khoa + "_" + thuat_toan
        }
        headers_old = Header.find_one(query={
                "identify": khoa + "_" + thuat_toan
            })
        if headers_old == -1:
            Header.insert(data=data)
        else:
            Header.update_one(query={"identify": khoa + "_" + thuat_toan},
                            change={"header": headers})
        return "okke"
    except Exception as e:
        print("err save_header_to_mongo: ", e)
        return "0"

def save_model(thuat_toan, khoa, classifier):
    try:
        pkl_filename = khoa + "_" + thuat_toan + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)
        return "okke"
    except Exception as e:
        print("err: ", e)
        return 0



def saved_score(score, model_name, maxx=[], minn=[]):
    score_saved = Score.find_one({
        "train_model": model_name
        })
    round_score = round(score, 2)
    if score_saved == -1:
        Score.insert(data= {
            "train_model": model_name,
            "score": round_score * 100,
            "max": maxx,
            "min": minn
        })
    else:
        Score.update_one(query= {"train_model": model_name},
                         change= {
                            "score": round_score * 100,
                            "max": maxx,
                            "min": minn
                         })
    return round_score