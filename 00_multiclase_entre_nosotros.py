import nltk
from nltk.metrics import agreement
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance

dataset1 = [
    ("facu", "1", frozenset(["Not hateful"])),
    ("juli", "1", frozenset(["Not hateful"])),
    ("facu", "2", frozenset(["Racism"])),
    ("juli", "2", frozenset(["Not hateful"])),
    ("facu", "3", frozenset(["Not hateful"])),
    ("juli", "3", frozenset(["Not hateful"])),
    ("facu", "4", frozenset(["Racism"])),
    ("juli", "4", frozenset(["Racism"])),
    ("facu", "5", frozenset(["Politics"])),
    ("juli", "5", frozenset(["Politics"])),
    ("facu", "6", frozenset(["Racism"])),
    ("juli", "6", frozenset(["Racism"])),
    ("facu", "7", frozenset(["Not hateful"])),
    ("juli", "7", frozenset(["Not hateful"])),
    ("facu", "8", frozenset(["Racism"])),
    ("juli", "8", frozenset(["Not hateful"])),
    ("facu", "9", frozenset(["Not hateful"])),
    ("juli", "9", frozenset(["Not hateful"])),
    ("facu", "10", frozenset(["Not hateful"])),
    ("juli", "10", frozenset(["Not hateful"])),
]

dataset2 = [
    ("facu", "1", frozenset(["Not hateful"])),
    ("juli", "1", frozenset(["Not hateful"])),
    ("facu", "2", frozenset(["Not hateful"])),
    ("juli", "2", frozenset(["Not hateful"])),
    ("facu", "3", frozenset(["Not hateful"])),
    ("juli", "3", frozenset(["Not hateful"])),
    ("facu", "4", frozenset(["Appearence"])),
    ("juli", "4", frozenset(["Not hateful"])),
    ("facu", "5", frozenset(["Not hateful"])),
    ("juli", "5", frozenset(["Not hateful"])),
    ("facu", "6", frozenset(["Not hateful"])),
    ("juli", "6", frozenset(["Not hateful"])),
    ("facu", "7", frozenset(["Not hateful"])),
    ("juli", "7", frozenset(["Not hateful"])),
    ("facu", "8", frozenset(["Appearence"])),
    ("juli", "8", frozenset(["Appearence"])),
    ("facu", "9", frozenset(["Politics"])),
    ("juli", "9", frozenset(["Politics"])),
    ("facu", "10", frozenset(["Politics", "Appearence"])),
    ("juli", "10", frozenset(["Politics", "Appearence"])),
]


task1 = AnnotationTask(distance=masi_distance)
task2 = AnnotationTask(distance=masi_distance)

task1.load_array(dataset1)
task2.load_array(dataset2)

print("IAA entre nosotros para el dataset 1 (multiclase):", task1.alpha())
print("IAA entre nosotros para el dataset 2 (multiclase):", task2.alpha())
