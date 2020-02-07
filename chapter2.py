#********************************************************************************************************
#                                Accessing Text Corpora and Lexical Resources
#********************************************************************************************************                                
'''
Accesing Text Corpora
nltk.corpus.gutenberg.fileids()

import nltk
Corpus = nltk.corpus.gutenberg.fileids()
print(Corpus)

Pick out the first of these texts

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
print( len(emma) )
emma = nltk.Text( nltk.corpus.gutenberg.words('austen-emma.txt') )
emma.concordance('surprize')

'''
import nltk
from nltk.corpus import gutenberg
gutenberg.fileids()
#print( gutenberg.fileids() )
emma = gutenberg.words('austen-emma.txt')
print(len(emma))

'''
This program displays three statistics for each text: average word length, average sentence 
length, and the number of times each vocabulary item appears in the text on average 
(our lexical diversity score).
'''

for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print (int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab),fileid )

#Returns a List of sentences
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')

print( macbeth_sentences )
print( macbeth_sentences[1037] )

#Return the max len of sentences 
longest_len = max([len(s) for s in macbeth_sentences])

#Save the sentences biggest
longest_sent = [s for s in macbeth_sentences if len(s) == longest_len]

#********************************************************************************************************
#                                        Web and Chat Text
#********************************************************************************************************

'''
Web Texts
'''
from nltk.corpus import webtext
for fileid in webtext.fileids():
    print fileid, webtext.raw(fileid)[:65]
'''
Chats
'''
from nltk.corpus import nps_chat
chatroom = nps_chat.posts('10-19-20s_706posts.xml')
chatroom[123]


#********************************************************************************************************
#                                        Brown Corpus
#********************************************************************************************************

from nltk.corpus import brown

Displsy the categories that it have.
brown.categories()

We can choose a especific genre and extract a list of words 
brown.words(categories='news')
brown.words(categories='cg22')


'''
'''
from nltk.corpus import brown

#Let's compare genres in their usage of modal verbs
news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print (m + ':', fdist[m])

editorial_text = brown.words(categories='editorial')
dfist = nltk.FreqDist([ w.lower() for w in editorial_text ])
wh = ['what', 'when', 'where', 'who','why']

for w in wh:
    print (w + ':', dfist[w])


#Conditional frequency distributions
cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']

#Make a table with the frequency in each genre
cfd.tabulate(conditions=genres, samples=modals)

#********************************************************************************************************
#                                        Reuters Corpus
#********************************************************************************************************
'''
from nltk.corpus import reuters
reuters.fileids()
'''

from nltk.corpus import reuters

#Show the ids of the categories
reuters.fileids()

#Show the categorie
reuters.categories()

#Show the categories in specific id
reuters.categories('training/9865')
reuters.categories(['training/9865', 'training/9880'])

#Show the text in a categorie
reuters.fileids('barley')

#we can specify the words or sentences we want in terms of files or categories
reuters.words('training/9865')[:14]
reuters.words(['training/9865', 'training/9880'])
reuters.words(categories='barley')
reuters.words(categories=['barley', 'corn'])

#********************************************************************************************************
#                                        Inaugural Address Corpus
#********************************************************************************************************
'''
from nltk.corpus import inaugural
inaugural.fileids()
'''

from nltk.corpus import inaugural
#inaugural.fileids()

cfd = nltk.ConditionalFreqDist( (target, fileid[:4])
           for fileid in inaugural.fileids()
           for w in inaugural.words(fileid)
           for target in ['america', 'citizen']
           if w.lower().startswith(target) )

cfd.plot()

#********************************************************************************************************
#                                        Corpora in Other Languages
#********************************************************************************************************

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
           (lang, len(word))
           for lang in languages
           for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)


#********************************************************************************************************
#                                            Loading your own Corpus
#********************************************************************************************************                                            
'''
If you have a your own collection you could use it, NLTK provides the folling option
'''
from nltk.corpus import PlaintextCorpusReader
#path of your collection
corpus_root = '/home/randy/Documentos/SelectedTopics/Corpus'
wordlists = PlaintextCorpusReader(corpus_root, '.*')

#Display the texts in the corpus selecte above
print( wordlists.fileids())

#Return a type NLTK so you need to change a List in Python if you can display all.
print(wordlists.words('e961024.htm'))

#wordlist = [word for word in wordlists.words('e961024.htm') ]
#print(wordlist)
'''
'''
You can use the follow for subfolders

corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj"
file_pattern = r".*/wsj_.*\.mrg"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()


#********************************************************************************************************
#                                        Condicional Frequency Distribution
#********************************************************************************************************

#Counting Words by Genre
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
           (genre, word)
           for genre in brown.categories()
           for word in brown.words(categories=genre))

#Especific categories
genre_word = [(genre, word)
               for genre in ['news', 'romance']
               for word in brown.words(categories=genre)]
len(genre_word)

print(genre_word[:4])
print(genre_word[-4:])

#Return te type and the number of conditions
print(cfd)

cfd = nltk.ConditionalFreqDist(genre_word)
#Return the conditions
print(cfd.conditions())


#********************************************************************************************************
#                                        Plotting and Tabulating Distributions
#********************************************************************************************************

from nltk.corpus import udhr
languages = ['Chickasaw', 'English', 'German_Deutsch', 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 
            'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
            (lang, len(word))
            for lang in languages
            for word in udhr.words(lang + '-Latin1'))

#Tabulating Distributions
# Samples = we can limit the samples to display with a samples= parameter
# We can optionally specify which conditions to display with a conditions= parameter.

cfd.tabulate(conditions=['English', 'German_Deutsch'],
              samples=range(10), cumulative=True)

#Generating Random Text with Bigrams

sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven',
        'and', 'the', 'earth', '.']

#Generate Biagrams and return a type generator object bigrams so we need to convert to List
nltk.bigrams(sent)

#biagrams = [ bi for bi in nltk.bigrams(sent)]
#print(biagrams)

'''
def generate_model(cfdist,word,num=15):
    for i in range(num):
        print(word + ' ' , end='')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams) 

print cfd['living']
generate_model(cfd, 'living')

'''
