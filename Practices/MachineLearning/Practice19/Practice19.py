import numpy as np
from mord import LogisticIT
from nltk import word_tokenize
from nltk.corpus import stopwords
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

'''
+---------------------------------------------------------------+
|                        Ordinal Logistic                       |
|                           Regression                          |
+---------------------------------------------------------------+
'''
lenSet = 3380

def read_texts( path ):
    comments = [ ]
    for i in range( 2 , lenSet ):
        comment = [ ]
        file = path + str(i) + '.review.pos'
        try:
            f = open( file , encoding = 'ISO-8859-1' )
            lines = [ line  for line in f ]
            tokens = [ word_tokenize(line) for line in lines ]
            comment = [ line[1] for line in tokens if len(line) > 0 ]
            comments.append( comment )
        except FileNotFoundError:
            print( 'File: {} not found!' .format(file) , end='\r')
    
    return comments

def clean_comments( comments ):
    c_comments = [ ]
    stop_words = stopwords.words('spanish')
    for comment in comments:
        clean_comment = [ token for token in comment if token not in stop_words and token.isalpha() ]
        c_comments.append( ' '.join(clean_comment) )
    
    return c_comments

def read_ranks( path ):
    ranks = [ ]
    for i in range( 2, lenSet ):
        file = path + str(i) + '.xml'
        try:
            f = open( file , encoding = 'ISO-8859-1' )
            line = f.readline()
            j = line.index( ' rank' )
            ranks.append( int( line[j+7] )  )
        except FileNotFoundError:
            print( 'File: {} not found!' .format(file) , end='\r' )

    return ranks

def tfidf_extractor( corpus ):
    vectorizer = TfidfVectorizer(norm='l2',smooth_idf=True,use_idf=True)
    features = vectorizer.fit_transform( corpus )
    return features 

if __name__ == "__main__":
    comments = read_texts( '/home/randy/Descargas/corpusCine/corpusCriticasCine/' )
    print('\n')
    comments = clean_comments( comments )
    ranks = read_ranks( '/home/randy/Descargas/corpusCine/corpusCriticasCine/' )
    print('\n')
    vectors = tfidf_extractor( comments )
    vectors = np.round( vectors.todense(), 2 )
    x_train, x_test, y_train, y_test = train_test_split(vectors,ranks,test_size=0.3,random_state=42)
    y_train = np.array( y_train )
    y_test = np.array( y_test )
    clf = LogisticIT()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    confused_matrix = confusion_matrix( y_test , y_pred )
    table = PrettyTable()
    table.field_names = [' ', '1' , '2','3','4','5']
    for i,row in enumerate(confused_matrix):
        aux = list( row )
        aux.insert(0,i+1)
        table.add_row( aux )
    print('''
    \t**************************************************
                      Matriz de confusion 
    \t**************************************************''')
    print( table )
    metrics = classification_report( y_test , y_pred , zero_division=1 )
    print('''
    \t**************************************************
                           Metricas 
    \t**************************************************''')
    print( metrics )
