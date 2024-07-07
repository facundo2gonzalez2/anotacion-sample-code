import pandas as pd
import krippendorff
import numpy as np


dataset1_nuestro = pd.read_csv("data/dataset1-nuestro.csv", dtype=int)
dataset1_anotadores = pd.read_csv("data/dataset1-anotadores.csv", dtype=int)
dataset1_chatgpt = pd.read_csv("data/dataset1-chatgpt.csv", dtype=int)

dataset1_binaria_nuestra = dataset1_nuestro["HATEFUL"].to_numpy()
dataset1_binaria_anotadores = dataset1_anotadores["HATEFUL"].to_numpy()
dataset1_binaria_chatgpt = dataset1_chatgpt["HATEFUL"].to_numpy()

dataset1_anotadores_chatgpt = []
dataset1_nuestro_chatgpt = []

for i in range(len(dataset1_binaria_nuestra)):
    dataset1_anotadores_chatgpt.append(
        [dataset1_binaria_anotadores[i], dataset1_binaria_chatgpt[i]])
    dataset1_nuestro_chatgpt.append(
        [dataset1_binaria_nuestra[i], dataset1_binaria_chatgpt[i]])

# Transforma los datos a una lista de listas para la función krippendorff.alpha
dataset1_nues = np.array(dataset1_nuestro_chatgpt).T.tolist()
dataset1_anot = np.array(dataset1_anotadores_chatgpt).T.tolist()

# Calcula el coeficiente alfa de Krippendorff
alpha1_anot = krippendorff.alpha(
    reliability_data=dataset1_anot, level_of_measurement="nominal")
alpha1_nues = krippendorff.alpha(
    reliability_data=dataset1_nues, level_of_measurement="nominal")

print(f"IAA para el dataset 1 chatgpt vs nosotros (binaria): {alpha1_nues}")
print(f"IAA para el dataset 1 chatgpt vs anotadores (binaria): {alpha1_anot}")
