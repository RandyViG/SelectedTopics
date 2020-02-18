import nltk
from nltk.book import *
from nltk.corpus import brown
from nltk.corpus import wordnet as wn

# Ejercicio 1
# Generar las graficas de la distribución acumulativa y de frecuencia de 1000 palabras del 
# vocabulario de text5
print('\n\n EJERCICIO 1 \n\n')
fdist1 = FreqDist(text5)
fdist1.plot(1000, cumulative=True)
fdist1.plot(1000)
# Ley de Zift:
# Muy pocas palabras con mucha frecuencia y muchas palabras con pocas frecuencia 

# Ejercicio 2
#Impirmir las palabras de text5 cuya longitud es > 30
print('\n\n EJERCICIO 2 \n\n')
V = set(text5)
long_words = [w for w in V if len(w) > 30]
for word in long_words:
    print(word)

# Ejercicio 3
# Genrara la distribución condicional para Brown Corpus e imprimir la tabla de frecuencia de 
# palabras: love, hate, speak, control, feel, great, president
# en los generos: news, romance, humor
print('\n\n EJERCICIO 3 \n\n')
cfd = nltk.ConditionalFreqDist( (genre, word) 
                                for genre in brown.categories() 
                                for word in brown.words(categories=genre) )
genres=['news', 'romance', 'humor']
modals=['love', 'hate', 'speak', 'control', 'feel', 'great', 'president']
cfd.tabulate(conditions=genres, samples=modals)


# Ejercicio 4
# Imprimir los sysnsets de palabras: computer, machine, car, sandwich
# Imprimir la similitud entre:
# computer y machine
# car y machine 
# car y computer
# computer y sandwich

print('\n\n EJERCICIO 4 \n\n')
computer = wn.synsets('computer')[0]
car = wn.synsets('car')[0]
machine = wn.synsets('machine')[0]
sandwich = wn.synsets('sandwich')[0]

print('Synset de Computer:')
print(computer)
print('Synset de Car:')
print(car)
print('Sysnset de machine')
print(machine)
print('Synset de sandwich')
print(sandwich)

print('Similitud entre computer y machine:')
print(computer.path_similarity(machine))
print('Similitud entre car y machine')
print(car.path_similarity(machine))
print('Similitud entre car y computer')
print(car.path_similarity(computer))
print('Similitud entre computer y sandwich')
print(computer.path_similarity(sandwich))







