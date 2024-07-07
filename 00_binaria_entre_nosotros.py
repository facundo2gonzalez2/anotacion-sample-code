import krippendorff
import numpy as np


dataset1 = np.array(
    [[0, 0], [1, 0], [0, 0], [1, 1], [1, 1], [
        1, 1], [0, 0], [1, 0], [0, 0], [0, 0]]
)

dataset2 = np.array(
    [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [
        0, 0], [0, 0], [1, 1], [1, 1], [1, 1]]
)


# Transforma los datos a una lista de listas para la función krippendorff.alpha
dataset1 = dataset1.T.tolist()

dataset2 = dataset2.T.tolist()

# Calcula el coeficiente alfa de Krippendorff
alpha1 = krippendorff.alpha(
    reliability_data=dataset1, level_of_measurement="nominal")
alpha2 = krippendorff.alpha(
    reliability_data=dataset2, level_of_measurement="nominal")

print(f"IAA entre anotadores para el dataset 1 (binaria): {alpha1}")
print(f"IAA entre anotadores para el dataset 2 (binaria): {alpha2}")
