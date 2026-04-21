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
        #Convertir texto a minúsculas
        texto = texto.lower()

        #Remover placeholders
        texto = re.sub(r'\{\{[^}]+\}\}', '', texto)
        texto = re.sub(r'\{[^}]+\}', '', texto)

        #Remover URLS
        texto = re.sub(r'http\S+|www\S+', '', texto)

        #Remover emails
        texto = re.sub(r'\S+@\S+', '', texto)

        #Remover numeros
        texto = re.sub(r'\d+', '', texto)

        #Remover puntuación
        texto = re.sub(r'[^\w\s]', ' ', texto)

        #Tokenizar texto
        tokens = word_tokenize(texto)

        # Eliminar stopwords
        tokens = [palabra for palabra in tokens if palabra not in self.stop_words]
    
        # Aplicar stemming
        tokens = [self.stemmer.stem(palabra) for palabra in tokens]
    
        # Eliminar tokens muy cortos
        tokens = [palabra for palabra in tokens if len(palabra) > 2]

        return tokens