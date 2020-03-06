from bs4 import BeautifulSoup

def openText():
    with open('../Corpus/e961024.htm', encoding = 'utf-8') as f:
        text = f.read()
        f.close()
        text = soupText( text )
    
    return text

def soupText( text ):
    soup = BeautifulSoup( text, 'lxml' ) 

    return soup

if __name__ == "__main__":
    tex = openText()
    titles = tex.findAll('h3')

    fv=open('TitleTopic.txt','w')
    for t in titles:
        fv.write( '{:30}\n'.format( str(t) ) )
        