from flask import Flask, render_template, request, jsonify
import joblib
import sys
import os
import json

#Agregar Backend al path para importar
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Backend'))

from preprocesar import ProcesadorTexto

app = Flask(__name__)

print("Cargando modelo de clasificación...")
try:
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'models')
    modelo = joblib.load(os.path.join(models_dir, 'modelo.pkl'))
    vocabulario = joblib.load(os.path.join(models_dir, 'vocabulario.pkl'))
    procesador = joblib.load(os.path.join(models_dir, 'preprocesador.pkl'))

    print(f"Modelo cargado exitosamente")
    print(f"Clase: {modelo.clases}")
    print(f"Vocabulario: {len(vocabulario)} palabras")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predecir():
    try:
        #Obtener datos del request
        data = request.get_json()
        texto = data.get('text', '')

        if not texto.strip():
            return jsonify({'error': 'El texto no puede estar vacío'}), 400
        
        #Preprocesar el texto
        tokens = procesador.limpiar_texto(texto)

        #Predecir
        categoria_pred = modelo.predecir([tokens])[0]
        probabilidades = modelo.predecir_probabilidades([tokens])[0]

        #Preparar respuesta
        response = {
            'categoria': categoria_pred,
            'confianza': probabilidades[categoria_pred],
            'probabilidades':{
                clase: round(prob, 3)
                for clase, prob in sorted(
                    probabilidades.items(),
                    key = lambda x: x[1],
                    reverse=True
                )
            },
            'tokens_procesados': len(tokens)
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/api/metrics')
def api_metrics():
    try:
        metrics_path = os.path.join(
            os.path.dirname(__file__), '..', 'Backend', 'models', 'metricas.json'
        )
        
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metricas = json.load(f)
        
        return jsonify(metricas)
    except FileNotFoundError:
        return jsonify({
            'error': 'Métricas no encontradas. Ejecuta train.py primero.'
        }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    if modelo is None:
        return jsonify({'status': 'error', 'message': 'Modelo no cargado'}), 500
    return jsonify({
    'status': 'ok', 
    'modelo': 'Naive Bayes',       
    'clases': modelo.clases,
    'vocabulario': len(vocabulario)        
    })

if __name__ == '__main__':
    if modelo is None:
        print("\n El modelo no se cargo correctamente. Verifique las rutas")
    
    print("Servidor flask iniciado")
    print("URL: http://localhost:5000")

    app.run(debug=True, port=5000)