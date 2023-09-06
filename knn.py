import collections
import numpy as np
import knn


def calcular_distancia(elemento, entrenamiento):
    elemento = np.asarray(elemento)
    entrenamiento = np.asarray(entrenamiento)
    distancias = []

    for elemento_entrenamiento in entrenamiento:
        suma = 0
        for j in range(len(elemento)):
            suma += (elemento[j] - elemento_entrenamiento[j]) ** 2
        distancia = suma ** 0.5
        distancias.append(distancia)
    return distancias

def ordenar_diccionario(diccionario):
    llaves = diccionario.keys()
    llaves_ordenadas = sorted(llaves)
    diccionario_ordenado = {}
    for llave in llaves_ordenadas:
        diccionario_ordenado[llave] = diccionario[llave]
    return diccionario_ordenado

def dividir_diccionario(diccionario, k):
    diccionario_dividido = {}
    for i, (clave, valor) in enumerate(diccionario.items()):
        if i < k:
            diccionario_dividido[clave] = valor
    return diccionario_dividido

def mas_comun(diccionario):
    valores = list(diccionario.values())
    contador = collections.Counter(valores).most_common(1)[0][0]
    return contador

def knn(elementos, entrenamiento, etiquetas, k=3):
    predicciones = []
    for elemento in elementos:
        distancias = calcular_distancia(elemento, entrenamiento)
        diccionario_distancias = dict(zip(distancias, etiquetas))  # se crea un diccionario de la etiqueta y distancia
        diccionario_distancias = ordenar_diccionario(diccionario_distancias)  # se ordenan
        diccionario_distancias = dividir_diccionario(diccionario_distancias,
                                                     k)  # se sobreescribe el diccionario con k elementos
        prediccion = mas_comun(diccionario_distancias)  # se elige el que más se repite
        predicciones.append(prediccion)
    return predicciones

def accuracy(predicciones, etiquetas):
    contador = 0
    for i in range(len(predicciones)):
        if predicciones[i] == etiquetas[i]:
            contador += 1
    return (contador / len(predicciones)) * 100


X_array = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
           [6.9, 3.1, 4.9, 1.5], [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1]]
tags = [1, 1, 1, 2, 2, 2, 3, 3, 3]
clasificar = [[5.9, 3, 5.1, 1.8], [9, 7.1, 5.9, 2.1]]

# knn(clasificar, X_array, tags, 3)
