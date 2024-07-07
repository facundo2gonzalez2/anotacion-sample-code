import pandas as pd
import krippendorff
import numpy as np


dataset1_nuestro = pd.read_csv("data/dataset1-nuestro.csv", dtype=int)
dataset1_anotadores = pd.read_csv("data/dataset1-anotadores.csv", dtype=int)
dataset2_nuestro = pd.read_csv("data/dataset2-nuestro.csv", dtype=int)
dataset2_anotadores = pd.read_csv("data/dataset2-anotadores.csv", dtype=int)


dataset1_binaria_nuestra = dataset1_nuestro["HATEFUL"].to_numpy()
dataset1_binaria_anotadores = dataset1_anotadores["HATEFUL"].to_numpy()
dataset2_binaria_nuestra = dataset2_nuestro["HATEFUL"].to_numpy()
dataset2_binaria_anotadores = dataset2_anotadores["HATEFUL"].to_numpy()

dataset1 = []
dataset2 = []

for i in range(len(dataset1_binaria_nuestra)):
    dataset1.append([dataset1_binaria_anotadores[i],
                    dataset1_binaria_nuestra[i]])
    dataset2.append([dataset2_binaria_anotadores[i],
                    dataset2_binaria_nuestra[i]])


dataset1 = np.array(dataset1).T.tolist()
dataset2 = np.array(dataset2).T.tolist()


alpha1 = krippendorff.alpha(
    reliability_data=dataset1, level_of_measurement="nominal")
alpha2 = krippendorff.alpha(
    reliability_data=dataset2, level_of_measurement="nominal")

print(f"IAA para el dataset 1 nosotros vs anotadores (binaria): {alpha1}")
print(f"IAA para el dataset 2 nosotros vs anotadores (binaria): {alpha2}")
