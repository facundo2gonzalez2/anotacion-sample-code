import pandas as pd
import krippendorff
import numpy as np


dataset2_nuestro = pd.read_csv("data/dataset2-nuestro.csv", dtype=int)
dataset2_anotadores = pd.read_csv("data/dataset2-anotadores.csv", dtype=int)
dataset2_chatgpt = pd.read_csv("data/dataset2-chatgpt.csv", dtype=int)

dataset2_binaria_nuestra = dataset2_nuestro["HATEFUL"].to_numpy()
dataset2_binaria_anotadores = dataset2_anotadores["HATEFUL"].to_numpy()
dataset2_binaria_chatgpt = dataset2_chatgpt["HATEFUL"].to_numpy()

dataset2_anotadores_chatgpt = []
dataset2_nuestro_chatgpt = []

for i in range(len(dataset2_binaria_nuestra)):
    dataset2_anotadores_chatgpt.append(
        [dataset2_binaria_anotadores[i], dataset2_binaria_chatgpt[i]])
    dataset2_nuestro_chatgpt.append(
        [dataset2_binaria_nuestra[i], dataset2_binaria_chatgpt[i]])

# Transforma los datos a una lista de listas para la función krippendorff.alpha
dataset2_nues = np.array(dataset2_nuestro_chatgpt).T.tolist()
dataset2_anot = np.array(dataset2_anotadores_chatgpt).T.tolist()

# Calcula el coeficiente alfa de Krippendorff
alpha1_anot = krippendorff.alpha(
    reliability_data=dataset2_anot, level_of_measurement="nominal")
alpha1_nues = krippendorff.alpha(
    reliability_data=dataset2_nues, level_of_measurement="nominal")

print(f"IAA para el dataset 2 chatgpt vs nosotros (binaria): {alpha1_nues}")
print(f"IAA para el dataset 2 chatgpt vs anotadores (binaria): {alpha1_anot}")
