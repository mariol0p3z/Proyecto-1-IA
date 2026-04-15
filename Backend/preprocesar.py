import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    print("Recursos de NLTK descargados correctamente")

class ProcesadorTexto:
    def __init__(self, language='english', min_word_length=2):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()
        self.min_word_length = min_word_length

    def limpiar_texto(self, texto):
        if not isinstance(texto, str):
            texto = str(texto)

        #Convertir texto a minúsculas
        texto = texto.lower()

        #Remover placeholders
        texto = re.sub(r'\{[^}]+\}', '', texto)

        #Remover URLS
        texto = re.sub(r'http\S+|www\S+', '', texto)

        #Remover emails
        texto = re.sub(r'\S+@\S+', '', texto)

        #Remover numeros
        texto = re.sub(r'\d+', '', texto)

        #Remover puntuación
        texto = texto.translate(str.maketrans('', '', string.punctuation))

        #Tokenizar texto
        tokens = word_tokenize(texto)

        #Remover stop words, palabras cortas y aplicar stemming
        tokens = [
            self.stemmer.stem(palabra)
            for palabra in tokens
            if palabra not in self.stop_words and len(palabra) > self.min_word_length
        ]

        return tokens
    
if __name__ == "__main__":
    procesador = ProcesadorTexto()

    ejemplos = [
        "I'm having an issue with the {product_purchased}. Please assist.",
        "I want to cancel my subscription immediately. Please process refund.",
        "How do I reset my password? I can't login to my account.",
        "I was charged twice for the same purchase. Need refund ASAP!",
        "What are your business hours? I have a question about pricing."
    ]

    print("Prueba del Preprocesador de Texto:")
    
    for i, ejemplo in enumerate(ejemplos, 1):
        tokens = procesador.limpiar_texto(ejemplo)
        print(f"\n--- Ejemplo {i} ---")
        print(f"Original: {ejemplo}")
        print(f"Tokens ({len(tokens)}): {tokens}")