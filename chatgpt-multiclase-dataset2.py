import pandas as pd
# import krippendorff
import numpy as np
import nltk
from nltk.metrics import agreement
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance


def extract_hateful_labels(row):
    possible_labels = [
        "WOMEN",
        "LGBTI",
        "RACISM",
        "CLASS",
        "POLITICS",
        "DISABLED",
        "APPEARANCE",
        "CRIMINAL",
    ]
    labels = []
    for label in possible_labels:
        if row[label]:
            labels.append(label)

    if labels == []:
        labels.append("NOT_HATEFUL")

    return frozenset(labels)


dataset2_nuestro = pd.read_csv("data/dataset2-nuestro.csv", dtype=int)
dataset2_anotadores = pd.read_csv("data/dataset2-anotadores.csv", dtype=int)
dataset2_chatgpt = pd.read_csv("data/dataset2-chatgpt.csv", dtype=int)

dataset2_nos = []
dataset2_anot = []


def add_rows_with_labels(extract_hateful_labels, labeledDataset, row1, row2):
    labeledDataset.append(
        ("comparacion", str(row1[0]), extract_hateful_labels(row1[1])))
    labeledDataset.append(
        ("chatgpt", str(row2[0]), extract_hateful_labels(row2[1])))


for row1, row2 in zip(dataset2_nuestro.iterrows(), dataset2_chatgpt.iterrows()):
    add_rows_with_labels(extract_hateful_labels, dataset2_nos, row1, row2)

for row1, row2 in zip(dataset2_anotadores.iterrows(), dataset2_chatgpt.iterrows()):
    add_rows_with_labels(extract_hateful_labels, dataset2_anot, row1, row2)


task1 = AnnotationTask(distance=masi_distance)
task2 = AnnotationTask(distance=masi_distance)

task1.load_array(dataset2_nos)
task2.load_array(dataset2_anot)

print("IAA para el dataset 2 chatgpt vs nosotros (multiclase):", task1.alpha())
print("IAA para el dataset 2 chatgpt vs anotadores (multiclase):", task2.alpha())
