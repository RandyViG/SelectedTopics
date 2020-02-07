#********************************************************************************************************
#                                      Processing Raw Text
#********************************************************************************************************

'''
you begin your interactive session or your program with the following import statements:
--> from __future__ import division
--> import nltk, re, pprint
'''
from __future__ import division
import nltk, re, pprint

#********************************************************************************************************
#                                      Electronics Books
#********************************************************************************************************

'''  In python3, urllib2 has been split into urllib.request and urllib.error 
url = "http://www.gutenberg.org/files/2554/2554.txt" Not Found!! '''
from urllib.request import urlopen
url = "http://www.gutenberg.org/files/49820/49820-0.txt"

''' Raw Function: Return bytes
Including many details we are not interested in such as whitespace, line breaks and blank lines. '''
raw = urlopen(url).read()
print('\n\nRaw text of: {}'.format(url) )
print( type(raw) )
print( len(raw) )
#raw = raw.decode() utf-8, ascii unicode-escape
raw = raw.decode('utf-8')
print( raw[:75] )

'''
If you're using an internet proxy which is not correctly detected by Python, you may need to specify the 
proxy manually as follows:
--> proxies = {'http': 'http://www.someproxy.com:3128'}
--> raw = urlopen(url, proxies=proxies).read()
'''

tokens = nltk.word_tokenize(raw)
print('\n\nTokenization')
print(tokens)
print( len(tokens) )
print( tokens[:10] )

'''Read the text like string with NLTK '''
text = nltk.Text(tokens)
print('\n\n')
print( type(text) )
#text = [ t for t in nltk.Text(tokens)]
#print( type(text) )
print(text[1020:1060])

'''This is a known bug. See NLTK issue #2299
print(text.collocations())'''

print('; '.join(text.collocation_list()))

print( raw.find("PART I") )
print( raw.rfind("End of Project Gutenberg's Crime") ) 
raw = raw[5303:1157681]
print( raw.find("PART I") )

#********************************************************************************************************
#                                          Dealing with HTML
#********************************************************************************************************

url = "http://news.bbc.co.uk/2/hi/health/2284783.stm"

html = urlopen(url).read()
html = html.decode('utf-8')

print( html[:60] )

from bs4 import BeautifulSoup
'''
Not work
raw = nltk.clean_html(html)'''

raw = BeautifulSoup(html,'lxml')
'''Need to change to text'''
text = raw.get_text()
tokens = nltk.word_tokenize(text)
print(tokens)

tokens = tokens[96:399]
text = nltk.Text(tokens)
text.concordance('gene')

#********************************************************************************************************
#                                         Processing RSS Feeds   
#********************************************************************************************************

import feedparser

llog = feedparser.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
llog['feed']['title']
print( len(llog.entries) )

post = llog.entries
print(type(post))

post = "".join([str(_) for _ in post])

print(post.title )
#content = post.content[0].value()
#print( content[:70] )

'''
nltk.word_tokenize(nltk.clean_html(content))
nltk.word_tokenize(nltk.clean_html(llog.entries[2].content[0].value))
'''
#raw = BeautifulSoup(content,'lxml')
#text = raw.get_text()

#********************************************************************************************************
#                                    Reading Local Files
#********************************************************************************************************

f = open('document.txt')
raw = f.read()

print(raw)

''' To check that the file that you are trying to open is really in the right directory '''
import os
print ( os.listdir('.') )

''' We can also read a file one line at a time  '''
for line in f:
        print ( line.strip() )

#********************************************************************************************************
#                                   Capturing User Input 
#********************************************************************************************************

s = input("Enter some text: ")
print ("You typed", len(nltk.word_tokenize(s)), "words.")

#********************************************************************************************************
#                              Extracting encoded text from files
#********************************************************************************************************

path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
import codecs

f = codecs.open(path, encoding='latin2')
for line in f:
        line = line.strip()
        print ( line.encode('unicode_escape') ) 

''' The module unicodedata lets us inspect the properties of Unicode characters '''

import unicodedata
lines = codecs.open(path, encoding='latin2').readlines()
line = lines[2]
print (line.encode('unicode_escape'))
for c in line:
        if ord(c) > 127:
                print ( '%r U+%04x %s' % (c.encode('utf8'), ord(c), unicodedata.name(c)) )

#********************************************************************************************************
#                              Using your local encoding in Python
#********************************************************************************************************

''' # -*- coding: <coding> -*- '''

#********************************************************************************************************
#                         Regular Expressions for Detecting Word Pattern
#********************************************************************************************************

import re
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]
print(wordlist)

''' Let's find words ending with ed using the regular expression «ed$» '''

print( [w for w in wordlist if re.search('ed$', w)] )

#********************************************************************************************************
#                                  Ranges and Closures
#********************************************************************************************************

print( [w for w in wordlist if re.search('^[ghi][mno][jlk][def]$', w)] )
print( [w for w in wordlist if re.search('^[g-i][m-o][j-k][d-f]$', w)] )
print( [w for w in wordlist if re.search('^[ghijklmno]+$', w)] )

chat_words = sorted(set(w for w in nltk.corpus.nps_chat.words()))
print( [w for w in chat_words if re.search('^m+i+n+e+$', w)] )
print ( [w for w in chat_words if re.search('^[ha]+$', w)] )

wsj = sorted(set(nltk.corpus.treebank.words()))

''' include de symbol .'''
print([w for w in wsj if re.search('^[0-9]+\.[0-9]+$', w)] )
''' include de symbol $ '''
print([w for w in wsj if re.search('^[A-Z]+\$$', w)] )
''' Range of numbers 0-0 except 4'''
print([w for w in wsj if re.search('^[0-9]{4}$', w)] )
''' Words with 3 or 5 letters '''
print([w for w in wsj if re.search('^[0-9]+-[a-z]{3,5}$', w)] )
'''Words with 5 letters or more, Words with 2 or 3 letters and Words with 6 or less letters'''
print([w for w in wsj if re.search('^[a-z]{5,}-[a-z]{2,3}-[a-z]{,6}$', w)] )
''' Words that end with ed or ing'''
print([w for w in wsj if re.search('(ed|ing)$', w)] ) 

#********************************************************************************************************
#                        Useful Applications of Regular Expressions
#********************************************************************************************************

word = 'supercalifragilisticexpialidocious'
print( re.findall(r'[aeiou]', word) )

''' Let's look for all sequences of two or more vowels in some text '''
wsj = sorted(set(nltk.corpus.treebank.words()))
fd = nltk.FreqDist(vs for word in wsj
                       for vs in re.findall(r'[aeiou]{2,}', word))
print(fd.items())

#********************************************************************************************************
#                               Searching Tokenized Text
#********************************************************************************************************

from nltk.corpus import gutenberg, nps_chat
moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
print( moby.findall(r"<a> (<.*>) <man>") )
chat = nltk.Text(nps_chat.words())
print( chat.findall(r"<.*> <.*> <bro>") )
print( chat.findall(r"<l.*>{3,}") )

from nltk.corpus import brown
hobbies_learned = nltk.Text(brown.words(categories=['hobbies', 'learned']))
print( hobbies_learned.findall(r"<\w*> <and> <other> <\w*s>") )

#********************************************************************************************************
#                           Regular Expressions for Tokenizing Text
#********************************************************************************************************

raw = '''When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's always pepper that makes people hot-tempered,'...'''

print(re.split(r' ', raw))
print(re.split(r'[ \t\n]+', raw))
''' W = [a-zA-Z0-9_] '''
print(re.split(r'\W+', raw))


''' The special (?x) "verbose flag" tells Python to strip out the embedded whitespace and comments. '''

text = 'That U.S.A. poster-print costs $12.40...'
pattern = r'''(?x)    # set flag to allow verbose regexps
     ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
   | \w+(-\w+)*        # words with optional internal hyphens
   | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
   | \.\.\.            # ellipsis
   | [][.,;"'?():-_`]  # these are separate tokens
 '''
print( nltk.regexp_tokenize(text, pattern) )

'''It will usually be necessary to wrap it so that it can be displayed conveniently. Consider the following 
output, which overflows its line, and which uses a complicated print statement
'''

saying = ['After', 'all', 'is', 'said', 'and', 'done', ',','more', 'is', 'said', 'than', 'done', '.']
for word in saying:
        print (word, '(' + str(len(word)) + '),')

from textwrap import fill
format = '%s (%d),'
pieces = [format % (word, len(word)) for word in saying]
output = ' '.join(pieces)
wrapped = fill(output)
print (wrapped)
