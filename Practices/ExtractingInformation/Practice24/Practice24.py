from nltk.corpus import cess_esp
from nltk import word_tokenize, sent_tokenize, DefaultTagger, RegexpTagger, UnigramTagger, RegexpParser

'''
+---------------------------------------------------------------+
|                     Information Extraction                    |
+---------------------------------------------------------------+
'''

lenSet = 25

def read_texts( path ):
    opinions = ' '
    for j in range( 1 , 6 ):
        for i in range( 1 , lenSet ):
            tokens = [ ]
            file_name = path + '_' + str(j) + '_' + str(i) + '.txt'
            try:
                f = open( file_name , encoding = 'ISO-8859-1' )
                lines = f.readlines()
                for line in lines:
                    opinion = ''.join(line)
                    opinions += opinion + ' '
            except:
                print( 'File: {} not found!' .format(file_name) , end='\r')
                continue
    print('\n')

    return opinions

def generateTagger():
    default_tagger = DefaultTagger('V')
    patterns = [(r'.*o$', 'NMS'),   # noun masculine singular
                (r'.*os$', 'NMP'),  # noun masculine plural
                (r'.*a$', 'NFS'),   # noun feminine singular
                (r'.*as$', 'NFP')   # noun feminine singular
                ]
    regexp_tagger = RegexpTagger(patterns, backoff=default_tagger)
    #train nltk.UnigramTagger using tagged sentences from cess_esp 
    cess_tagged_sents = cess_esp.tagged_sents()
    combined_tagger = UnigramTagger(cess_tagged_sents, backoff=regexp_tagger)
    
    return combined_tagger

def tag( tagger , tokens ):
    return tagger.tag( tokens )

if __name__ == '__main__':
    op_no = read_texts( '../../Corpus/peliculas/no' )
    op_yes = read_texts( '../../Corpus/peliculas/yes' )
    tagger = generateTagger()
    opinions = op_no + op_yes
    opinions = sent_tokenize( opinions , 'spanish' )
    tokens_list = [ ]
    for opinion in opinions:
        tokens_list.append( tag( tagger , word_tokenize(opinion) ) )
    with open('Opinions_tag.txt','w') as f:
        for tokens in tokens_list:
            for tok in tokens:
                f.write(tok[0] + '/' + tok[1] + ' ')
            f.write('\n')
        f.close()
    grammar = 'NP: {<da.*>?<nc.*><sp.*|p.*>*<V|v.*|N.*|n.*>+}'
    cp = RegexpParser( grammar )
    
    with open('Chunks.txt','w') as f:
        for tokens in tokens_list:
            tree = cp.parse( tokens )
            for subtree in tree.subtrees():
                if subtree.label() == 'NP':
                    f.write(str(subtree) + '\n')
        f.close()
    
    grammar_relation = '''
    NP: {<da.*>?<nc.*><sp.*|p.*>*<V|v.*|N.*|n.*>+}
    DE: {<sps00>}
    CHUNK: {<NP><DE><NP>} 
    '''
    cp = RegexpParser( grammar_relation )

    cont = 0
    with open('Pertenencia_Chunks.txt','w') as f:
        for tokens in tokens_list:
            tree = cp.parse( tokens )
            for subtree in tree.subtrees():
                if subtree.label() == 'CHUNK':
                    f.write(str(subtree) + '\n')
                    cont += 1
        f.close()

    print('Chunks encontrados: {}'.format(cont))