'''
f = open('Corpus/e961024.htm', encoding='utf-8')
textString = f.read()
print(type(textString))
print(len(textString))
print(textString[:1000])

import nltk
tokens = nltk.word_tokenize(textString)

text = nltk.Text(tokens)

print('This is the first words of the text')
print(text[:100])


Lectura de texto como cadena 
            |
            |
            --> Limpiar HTML
                    |
                    |
                    --> Tokenizar:
                            |    - word_tokenizer()
                            |    - PlaintextCorpusReader
                            |    - Split ()
                            |
                            --> Seleccion de los tokens relevantes para mi objetivo

'''
import nltk
from bs4 import BeautifulSoup
#Lectura
f = open('Corpus/e961024.htm', encoding='utf-8')
text = f.read()
f.close()
#Limpiar HTML
soup = BeautifulSoup(text,'lxml')
text = soup.get_text()
#Tokenizar
words = nltk.word_tokenize(text)
print(words)

