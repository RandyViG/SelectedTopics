from nltk.book import text3

vocabulary = sorted(set(text3))

print('Num de Palabras: {}'.format(len(vocabulary)) )
with open('vocabularyText3.txt', 'w') as f:
    for word in vocabulary:
        f.write(word+'\n')
    f.close()