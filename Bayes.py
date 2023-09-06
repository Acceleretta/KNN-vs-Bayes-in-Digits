import numpy as np
import statistics as stats
import math
from collections import Counter


def Bayes(elementos, entrenamiento, etiquetas):
    # tmp = elementos[etiquetas==0]
    num_clases = len(set(etiquetas))
    num_atributos = len(entrenamiento[0])
    predicciones = []
    diccionario_media = {i: [] for i in range(num_clases)}  # este diccionario va a almacenar la media y la dsv std
    diccionario_dsv_std = {i: [] for i in range(num_clases)}  # diccionario de dvs std
    columnas = crear_lista_columnas(entrenamiento, num_atributos)
    for columna in columnas:
        for clase in range(num_clases):
            media_clases(diccionario_media, columna, etiquetas, clase)
            dsv_std_clases(diccionario_dsv_std, columna, etiquetas, clase)
    for elemento in elementos:
        proba_clase = []
        for clase in range(num_clases):
            prediccion(elemento, diccionario_media, diccionario_dsv_std, clase, proba_clase, num_clases)
        predicciones.append(proba_clase.index(max(proba_clase)))
    return predicciones


def media_clases(diccionario_media, columna, etiquetas, clase):
    media_atributo = []
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta == clase:
            media_atributo.append(columna[i])
    media = stats.mean(media_atributo)
    diccionario_media[clase].append(media)


def dsv_std_clases(diccionario_dsv_std, columna, etiquetas, clase):
    dsv_std_atributo = []
    for i, etiqueta in enumerate(etiquetas):
        if etiqueta == clase:
            dsv_std_atributo.append(columna[i])
    dsv_std = np.std(dsv_std_atributo, ddof=0)
    if dsv_std == 0:
        dsv_std = 1e-10
    diccionario_dsv_std[clase].append(dsv_std)


def prediccion(elemento, media, dsv_std, clase, predicciones, num_clases):
    proba_clase = 1
    for atributo in range(len(elemento)):
        proba_clase *= 1 / ((2 * math.pi * dsv_std[clase][atributo]) ** 0.5) * math.exp(
            (-0.5) * ((elemento[atributo] - media[clase][atributo]) / dsv_std[clase][atributo]) ** 2)
    proba_clase *= 1 / num_clases
    predicciones.append(proba_clase)


def crear_lista_columnas(entrenamiento, num_columnas):
    columnas = []
    for i in range(num_columnas):
        columna_i = []
        for fila in entrenamiento:
            columna_i.append(fila[i])
        columnas.append(columna_i)
    return columnas


def accuracy(predicciones, etiquetas):
    contador = 0
    for i in range(len(predicciones)):
        if predicciones[i] == etiquetas[i]:
            contador += 1
    return (contador / len(predicciones)) * 100


def proporcion(etiquetas):
    contador = Counter(etiquetas)
    return dict(contador)


X_array = [[5.1, 3.5, 1.4, 0.2], [4.9, 3, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
           [6.9, 3.1, 4.9, 1.5], [6.3, 3.3, 6, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3, 5.9, 2.1]]
tags = [0, 0, 0, 1, 1, 1, 2, 2, 2]
clasificar = [[5.9, 3, 5.1, 1.8], [9, 7.1, 5.9, 2.1]]

Bayes(clasificar, X_array, tags)
