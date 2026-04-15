import numpy as np
from collections import defaultdict, Counter
import math

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.clases = []
        self.probabilidades_clase = {}
        self.conteo_palabras = {}
        self.total_palabras_clase = {}
        self.vocabulario = set()

    def entrenar(self, X, y):
        n_muestras = len(X)
        self.clases = list(set(y))

        conteo_clases = Counter(y)
        
        #Calcular probabilidades de cada clase: P(clase)
        for clase in self.clases:
            self.probabilidades_clase[clase] = conteo_clases[clase] / n_muestras

        #Contar palabras por clase
        for tokens, clase in zip(X, y):
            for palabra in tokens:
                self.vocabulario.add(palabra)
                if clase not in self.conteo_palabras:
                    self.conteo_palabras[clase] = {}
                if palabra not in self.conteo_palabras[clase]:
                    self.conteo_palabras[clase][palabra] = 0
                self.conteo_palabras[clase][palabra] += 1
        
        #Contar total de palabras por clase
        for clase in self.clases:
            self.total_palabras_clase[clase] = sum(self.conteo_palabras[clase].values())

        print("Modelo Entrenado:")
        print(f"Clases: {len(self.clases)}")
        print(f"Vocabulario: {len(self.vocabulario)} palabras")
        print(f"Muestras de Entrenamiento: {n_muestras}")

    def calcular_log_verosimilitud(self, palabra, clase):
        """Calcular P(palabra|clase) con Laplace Smoothing
        
        Formula:
        P(palabra|clase) = (conteo_palabra_clase + alpha) / (total_palabras_clase + alpha * |Vocabulario|)
        """
        conteo_palabra = self.conteo_palabras[clase].get(palabra, 0)
        total_palabras = self.total_palabras_clase[clase]
        tamanio_vocabulario = len(self.vocabulario)

        #Laplace Smoothing
        probabilidad = (conteo_palabra + self.alpha) / (total_palabras + self.alpha * tamanio_vocabulario)

        return math.log(probabilidad)
    
    def predecir(self, X):
        #Predice la clase para nuevos textos
        return [self.predecir_uno(tokens) for tokens in X]
    
    def predecir_uno(self, tokens):
        """Predecir la clase para un solo texto
        
        Formula:
        clase_predicha = argmax[log P(clase) + sum(log P(palabra|clase))]
        """
        puntajes_clase = {}

        for clase in self.clases:
            #Log P(clase)
            puntaje = math.log(self.probabilidades_clase[clase])

            #Sumar log P(palabra|clase) para cada palabra
            for palabra in tokens:
                if palabra in self.vocabulario:
                    puntaje += self.calcular_log_verosimilitud(palabra, clase)
            
            puntajes_clase[clase] = puntaje
        
        #Retornar la clase con mayor puntaje
        return max(puntajes_clase, key=puntajes_clase.get)
    
    def predecir_probabilidades(self, X):
        #Predice las probabilidades por clase para cada texto
        return [self.predecir_probabilidades_uno(tokens) for tokens in X]
    
    def predecir_probabilidades_uno(self, tokens):
        """Calcula probabilidades para un solo texto
        
        Convierte log scores a probabilidades normalizadas
        """
        puntajes_clase = {}
        for clase in self.clases:
            puntaje = math.log(self.probabilidades_clase[clase])
            for palabra in tokens:
                if palabra in self.vocabulario:
                    puntaje += self.calcular_log_verosimilitud(palabra, clase)
            puntajes_clase[clase] = puntaje

        #Convertir log scores a probabilidades
        #Restar el maximo para evitar overflow
        max_puntaje = max(puntajes_clase.values())
        puntajes_exp = {clase: math.exp(puntaje - max_puntaje) for clase, puntaje in puntajes_clase.items()}

        total = sum(puntajes_exp.values())

        return {clase: puntaje_exp / total for clase, puntaje_exp in puntajes_exp.items()}
    
# ===== PRUEBAS =====
if __name__ == "__main__":
    print("="*70)
    print("PRUEBA DEL CLASIFICADOR NAÏVE BAYES")
    print("="*70)
    
    # Datos de entrenamiento de ejemplo
    X_entrenamiento = [
        ['password', 'reset', 'help', 'account'],
        ['cant', 'login', 'access', 'account'],
        ['refund', 'charge', 'billing', 'money'],
        ['invoice', 'payment', 'billing', 'charge'],
        ['question', 'product', 'info', 'details'],
        ['cancel', 'subscription', 'stop', 'service'],
        ['want', 'refund', 'money', 'back'],
        ['technical', 'problem', 'not', 'working']
    ]
    
    y_entrenamiento = [
        'Technical issue',
        'Technical issue',
        'Billing inquiry',
        'Billing inquiry',
        'Product inquiry',
        'Cancellation request',
        'Refund request',
        'Technical issue'
    ]
    
    # Entrenar modelo
    print("\n--- ENTRENAMIENTO ---")
    clasificador = NaiveBayes(alpha=1.0)
    clasificador.entrenar(X_entrenamiento, y_entrenamiento)
    
    # Datos de prueba
    X_prueba = [
        ['password', 'reset', 'forgot'],
        ['refund', 'need', 'money'],
        ['cancel', 'subscription'],
        ['product', 'question', 'info'],
        ['billing', 'charge', 'wrong']
    ]
    
    print("\n--- PREDICCIONES ---")
    predicciones = clasificador.predecir(X_prueba)
    probabilidades = clasificador.predecir_probabilidades(X_prueba)
    
    for i, (tokens, pred, proba) in enumerate(zip(X_prueba, predicciones, probabilidades), 1):
        print(f"\nEjemplo {i}:")
        print(f"  Tokens: {tokens}")
        print(f"  Predicción: {pred}")
        print(f"  Probabilidades:")
        for clase, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True):
            print(f"    {clase}: {prob:.3f}")