#********************************************************************************************************
#                                Language Processing And Python
#********************************************************************************************************
'''
Requirements
-- Python 3
 
-- NLTK
    pip install --user -U nltk
'''
#********************************************************************************************************
#                                  Getting Started with NLTK
#********************************************************************************************************
'''
import nltk
nltk.download()
-- all
'''
#from NLTK’s book module, load all items.
from nltk.book import *

print("To know about text1: text1")
print(text1)

#********************************************************************************************************
#                                       Serching text
#********************************************************************************************************
'''
concordance view shows us every occurrence of a given word, together with some
context    
text1.concordance('monstrous')
'''
print('Text1: concordance monstrous')
text1.concordance('monstrous')
print('Text2: concordance affection')
text2.concordance('affection')
print('Text3: concordance lived')
text3.concordance('lived')

'''
words appear in a similar range of contexts
text1.similar('montrous')
'''
print('Text1: similar monstrous')
text1.similar('monstrous')
print('Text2: similar monstrous')
text2.similar('monstrous')

''' common_contexts allows us to examine just the contexts that are shared by
    two or more words, text2.common_contexts(["monstrous", "very"]) '''

print('Text2: common_contexts monstrous and very ')
text2.common_contexts( ['monstrous', 'very'] )

#********************************************************************************************************
#                                     Dispersion Plot
#********************************************************************************************************

text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])

''' Generate generating some random text in the various styles we have just seen. '''
text3.generate()

#********************************************************************************************************
#                                   Counting Vocabulary
#********************************************************************************************************

print('Number of words in text3: {}'.format(len(text3))) 
print('Make a vocabulary with set and sorted')
print( sorted(set(text3)) )

#Average of words are used on the text
print('Avarege in text3:')
print( len(text3) / len( set(text3) ) )

#Count a specific word
print('Count how often a smote occurs')
text3.count('smote')

#How often occurs in a text percentage
print( 100 * text4.count('a') / len(text4) )

print('Count how often a lol occurs %')
print( 100 * text5.count('lol') / len(text4) )

'''
def lexical_diversity(text):
    return len(text) / len(set(text))
    
def percentage(count, total):
    return 100 * count / total
'''

#********************************************************************************************************
#                                          Lists
#********************************************************************************************************
sent2 = ['The', 'family', 'of', 'Dashwood', 'had', 'long','been', 'settled', 'in', 'Sussex', '.']
print(sent2[0])
sent3 = ['In', 'the', 'beginning', 'God', 'created', 'the','heaven', 'and', 'the', 'earth', '.']
print(sent3)

sent3.append('some')
print(sent3)

#********************************************************************************************************
#                                   Frequency Distributions
#********************************************************************************************************
'''FreqDist() Return Diccionario de tuplas'''
fdist1 = FreqDist(text1)
print(fdist1)

vocabulary1 = fdist1.keys()

'''Produce un error
# vocabulary1[:50] '''

import operator

sorted_fdist1 = sorted( fdist1.items(), key = operator.itemgetter(1), reverse = True)
most_frequent = []
for i in range( len(sorted(fdist1)) ):
    most_frequent.append(sorted_fdist1[i][0])

first_most_frequent = most_frequent[:50]
print('First most frequent words: ')
print( first_most_frequent )

print('First most frequent words afetr sorting alphabetically: ')
print( sorted(first_most_frequent) )

fdist1.plot(50, cumulative=True)

#The words that occur once only
print( fdist1.hapaxes() )

#********************************************************************************************************
#                                  Fine-Grained Selection of Words
#********************************************************************************************************
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)

print(long_words)

fdist5 = FreqDist(text5)
sorted([w for w in set(text5) if len(w) > 7 and fdist5[w] > 7])

#********************************************************************************************************
#                                     Collocations and Bigrams
#********************************************************************************************************
print('\nCollocations')
# text4.collocations() it doesn't work yo need to use: collocation_list()
print(text4.collocation_list())
