import nltk
from pickle import dump, load
from bs4   import BeautifulSoup
from nltk.corpus import cess_esp

def get_sentences(fname):
    f=open(fname, encoding = 'utf-8')
    t=f.read()
    soup = BeautifulSoup(t, 'lxml')
    text_string = soup.get_text()

    #get a list of sentences
    sent_tokenizer=nltk.data.load('nltk:tokenizers/punkt/english.pickle')
    sentences=sent_tokenizer.tokenize(text_string)
    return sentences

def tag_with_pos_tag_function_nltk(sentences):
    for s in sentences:
        tokens=nltk.word_tokenize(s)
        s_tagged=nltk.pos_tag(tokens)
        print(s_tagged)

def make_and_save_default_tagger(fname, tag):
    default_tagger=nltk.DefaultTagger(tag)    
    
    output=open(fname, 'wb')
    dump(default_tagger, output, -1)
    output.close()
    return fname

def tag(fname, sentences):
    input=open(fname, 'rb')
    default_tagger=load(input)
    input.close()

    for s in sentences:
        tokens=nltk.word_tokenize(s)
        s_tagged=default_tagger.tag(tokens)
        print(s_tagged)

def make_and_save_regexp_tagger(fname):
    patterns=[ (r'.*o$', 'NMS'), # noun masculine singular
               (r'.*os$', 'NMP'), # noun masculine plural
               (r'.*a$', 'NFS'),  # noun feminine singular
               (r'.*as$', 'NFP')  # noun feminine singular
             ]
    regexp_tagger=nltk.RegexpTagger(patterns)    
    
    output=open(fname, 'wb')
    dump(regexp_tagger, output, -1)
    output.close()
    
    return fname

def make_and_save_most_common_words_lookup_tagger(fname, number):
    fd_words = nltk.FreqDist(cess_esp.words())
    fd_tagged_words = nltk.ConditionalFreqDist(cess_esp.tagged_words())
   
    most_common_words = fd_words.most_common(number)
    most_common_words = [item[0] for item in most_common_words]
    likely_tags = dict((word, fd_tagged_words[word].max()) for word in most_common_words) 
    lookup_tagger = nltk.UnigramTagger(model=likely_tags) 
    
    output=open(fname, 'wb')
    dump(lookup_tagger, output, -1)
    output.close()
        
def make_and_save_lookup_tagger(fname):
    fd_tagged_words = nltk.ConditionalFreqDist(cess_esp.tagged_words())
   
    likely_tags = dict((word, fd_tagged_words[word].max()) for word in cess_esp.words()) 
    lookup_tagger = nltk.UnigramTagger(model=likely_tags) 
    
    output=open(fname, 'wb')
    dump(lookup_tagger, output, -1)
    output.close()
        
def train_and_save_unigram_tagger(fname):
    #train nltk.UnigramTagger using
    #tagged sentences from cess_esp 
    cess_tagged_sents=cess_esp.tagged_sents()
    tagger=nltk.UnigramTagger(cess_tagged_sents)
    
    #save the trained tagger in a file
    output=open(fname, 'wb')
    dump(tagger, output, -1)
    output.close()

def make_and_save_combined_tagger(fname):
    default_tagger=nltk.DefaultTagger('V')
    patterns=[ (r'.*o$', 'NMS'), # noun masculine singular
               (r'.*os$', 'NMP'), # noun masculine plural
               (r'.*a$', 'NFS'),  # noun feminine singular
               (r'.*as$', 'NFP')  # noun feminine singular
             ]
    regexp_tagger=nltk.RegexpTagger(patterns, backoff=default_tagger)
    #train nltk.UnigramTagger using tagged sentences from cess_esp 
    cess_tagged_sents=cess_esp.tagged_sents()
    combined_tagger=nltk.UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    #save the trained tagger in a file
    output=open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()

if __name__=='__main__':
    fname='Corpus/e961024.htm'
    sentences=get_sentences(fname)
    sentence=sentences[12:13]
    print('\n', sentence, '\n' )
        
    print('\nTagged with NLTK pos_tag function:\n')
    tag_with_pos_tag_function_nltk(sentence)
    
    '''
    print('\nTagged with default tagger:\n')
    t = 'N'
    fname_default_tagger = 'default_tagger_' + t + '.pkl'
    #make_and_save_default_tagger(fname_default_tagger, tag)
    tag(fname_default_tagger, sentence)
    
    print('\nTagged with regexp tagger:\n')
    fname_regexp_tagger = 'regexp_tagger.pkl'
    #make_and_save_regexp_tagger(fname_regexp_tagger)
    tag(fname_regexp_tagger, sentence)
    
    print('\nTagged with most common words lookup tagger:\n')
    number_of_most_common_words = 1000
    fname_most_common_words_lookup_tagger = str(number_of_most_common_words) + \
                                            '_most_common_words_lookup_tagger.pkl' 
    #make_and_save_most_common_words_lookup_tagger(fname_most_common_words_lookup_tagger,
                                                  #number_of_most_common_words)
    tag(fname_most_common_words_lookup_tagger, sentence)
    
    print('\nTagged with lookup tagger:\n')
    fname_lookup_tagger = 'lookup_tagger.pkl' 
    #make_and_save_lookup_tagger(fname_lookup_tagger)
    tag(fname_lookup_tagger, sentence)
    
    print('\nTagged with unigram tagger:\n')
    fname_unigram_tagger = 'unigram_tagger.pkl' 
    #train_and_save_unigram_tagger(fname_unigram_tagger)
    tag(fname_unigram_tagger, sentence)
    
    print('\nTagged with combined tagger:\n')
    fname_combined_tagger = 'combined_tagger.pkl' 
    #make_and_save_combined_tagger(fname_combined_tagger)
    tag(fname_combined_tagger, sentence)
    
    '''
    ########################################################
    '''
    print()
    example_1=['Sal de la ciudad rapidamente.']
    tag('combined_tagger.pkl', example_1)
    print()
    example_2=['Sal de México es buena.']
    tag('combined_tagger.pkl', example_2)
    print()
    example_3=['El estado de la cuestión es no aceptable.']
    tag('combined_tagger.pkl', example_3)
    print()
    example_4=['Hemos estado en México varios días.']
    tag('combined_tagger.pkl', example_4)
    '''
 
    #########################################################
    '''
    NNP proper noun singular
    NNS common noun plural
    VB  verb, base form
    VBZ verb, present tense, 3rd person singular
    VBD verb, past tense
    JJ adjective or numeral
    NN common noun, singular or mass
    IN preposition or conjunction
    FW foreign word
    PDT pre-determiner (e.g. all, both, many, such, this)
    DT determiner (e.g. another, each, no, some, all, both, this, these)
    
    '''
    