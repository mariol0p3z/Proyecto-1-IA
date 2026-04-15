import pandas as pd
import numpy as np
import joblib
import os
from preprocesar import ProcesadorTexto
from naive_bayes import NaiveBayes
from metricas import calcular_metricas, matriz_confusion, imprimir_metricas, imprimir_matriz_confusion
from kfolds import dividir_kfolds, obtener_datos_fold

from config import(
    dataset,
    columna_categorica,
    columna_texto,
    categorias,
    laplace_alpha,
    k_folds,
    semilla,
    models_dir
)

def cargar_datos(ruta):
    print(f"Cargando datos desde: {ruta}")
    df = pd.read_csv(ruta)
    print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
    print(f"Distribucion de categorias:")
    print(df[columna_categorica].value_counts())
    return df

def procesar_dataset(df, columna_texto):
    procesador = ProcesadorTexto()

    x = []
    total = len(df)

    for i, texto in enumerate(df[columna_texto], 1):
        if i %1000 == 0:
            print(f"Procesados: {i}/{total}")
        
        tokens = procesador.limpiar_texto(texto)
        x.append(tokens)

    print(f"{len(x)} textos procesados")

    longitudes = [len(tokens) for tokens in x]
    print(f"Longitud promedio: {np.mean(longitudes):.1f} tokens")
    print(f"Longitud minima: {np.min(longitudes)} tokens")
    print(f"Longitud maxima: {np.max(longitudes)} tokens")

    print("\n🔍 DEBUG: Primeros 5 textos procesados por categoría:")
    for categoria in categorias:
        indices = df[df[columna_categorica] == categoria].index[:2]
        print(f"\n{categoria}:")
        for idx in indices:
            tokens = x[idx]
            print(f"  {tokens[:15]}...")  # Primeros 15 tokens
    # ===== FIN DEBUG =====
    
    return x, procesador

def ejecutar_kfolds(X, y, k =5):
    folds = dividir_kfolds(X, y, k=k, shuffle=True, semilla=semilla)

    metricas_folds = []
    matrices_folds = []

    for i, (indices_train, indices_test) in enumerate(folds, 1):
        X_train, X_test, y_train, y_test = obtener_datos_fold(X, y, indices_train, indices_test)

        print(f"Train: {len(X_train)} muestras")
        print(f"Test: {len(X_test)} muestras")

        modelo = NaiveBayes(alpha=laplace_alpha)
        modelo.entrenar(X_train, y_train)

        y_pred = modelo.predecir(X_test)

        # ===== DEBUG: Agregar estas líneas =====
        print(f"\n🔍 DEBUG Fold {i}:")
        print(f"  Clases que el modelo aprendió: {modelo.clases}")
        print(f"  Clases en config.py: {categorias}")
        print(f"  Primeras 10 predicciones: {y_pred[:10]}")
        print(f"  Primeras 10 reales: {list(y_test[:10])}")
        print(f"  ¿Coinciden tipos? y_pred[0]={type(y_pred[0])}, y_test[0]={type(y_test[0])}")
        print(f"  ¿Son iguales? {y_pred[0] == y_test[0]}")
        # ===== FIN DEBUG =====
        
        metricas = calcular_metricas(y_test, y_pred, categorias)
        matriz = matriz_confusion(y_test, y_pred, categorias)

        metricas_folds.append(metricas)
        matrices_folds.append(matriz)

        print(f"Accuracy: {metricas['accuracy']:.3f}")
        print(f"Macro F1-Score: {metricas['macro_f1']:.3f}")

    return metricas_folds, matrices_folds

def calcular_metricas_promedio(metricas_folds):
    accuracy_promedio = np.mean([m['accuracy'] for m in metricas_folds])
    macro_f1_promedio = np.mean([m['macro_f1'] for m in metricas_folds])

    accuracy_std = np.std([m['accuracy'] for m in metricas_folds])
    macro_f1_std = np.std([m['macro_f1'] for m in metricas_folds])

    #Promediar metricas por clase
    metricas_promedio = {}
    for categoria in categorias:
        precision_vals = [m[categoria]['precision'] for m in metricas_folds]
        recall_vals = [m[categoria]['recall'] for m in metricas_folds]
        f1_vals = [m[categoria]['f1_score'] for m in metricas_folds]

        metricas_promedio[categoria] = {
            'precision': np.mean(precision_vals),
            'recall': np.mean(recall_vals),
            'f1_score': np.mean(f1_vals),
            'support': metricas_folds[0][categoria]['support']  
        }

    metricas_promedio['accuracy'] = accuracy_promedio
    metricas_promedio['macro_f1'] = macro_f1_promedio

    #Imprimir resultados
    imprimir_metricas(metricas_promedio, categorias)

    print(f"\nVariabilidad entre folds:")
    print(f"Accuracy: {accuracy_promedio:.3f} ± {accuracy_std:.3f}")
    print(f"Macro F1-Score: {macro_f1_promedio:.3f} ± {macro_f1_std:.3f}")

    return metricas_promedio

def entrenar_modelo_final(X, y):
    modelo_final = NaiveBayes(alpha=laplace_alpha)
    modelo_final.entrenar(X, y)

    print(f"\n Modelo final entrando con {len(X)} muestras")

    return modelo_final

def guardar_modelo(modelo, procesador, directorio = models_dir):
    #Crear directorio si no existe
    os.makedirs(directorio, exist_ok=True)

    #Ruta de archivos
    ruta_modelo = os.path.join(directorio, 'modelo.pkl')
    ruta_vocab = os.path.join(directorio, 'vocabulario.pkl')
    ruta_prep = os.path.join(directorio, 'preprocesador.pkl')

    #Guardar
    joblib.dump(modelo, ruta_modelo)
    joblib.dump(modelo.vocabulario, ruta_vocab)
    joblib.dump(procesador, ruta_prep)

    print(f"Modelo guardado en: {ruta_modelo}")
    print(f"Vocabulario guardado en: {ruta_vocab}")
    print(f"Preprocesador guardado en: {ruta_prep}")

    #Tamaños de archivos
    tamanio_modelo = os.path.getsize(ruta_modelo) / 1024
    tamanio_vocab = os.path.getsize(ruta_vocab) / 1024
    print(f"Tamaño del modelo: {tamanio_modelo:.2f} KB")
    print(f"Tamaño del vocabulario: {tamanio_vocab:.2f} KB")

def main():
    print("Clasificador de tickets de soporte técnico - Naive Bayes")
    print("Entrenamiento con K-Folds Cross-Validation")

    df = cargar_datos(dataset)

    X, procesador = procesar_dataset(df, columna_texto)
    y = df[columna_categorica].values
    metricas_folds, matrices_folds = ejecutar_kfolds(X, y, k=k_folds)
    metricas_promedio = calcular_metricas_promedio(metricas_folds)
    modelo_final = entrenar_modelo_final(X, y)
    guardar_modelo(modelo_final, procesador)

    print("\nEntrenamiento finalizado")
    print("Resultados:")
    print(f"Accuracy promedio: {metricas_promedio['accuracy']:.3f}")
    print(f"Macro F1-Score promedio: {metricas_promedio['macro_f1']:.3f}")
    print(f"Modelo guardado en: {models_dir}")

if __name__ == "__main__":
    main()