import sys
#please install NLTK
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import numpy as np
from io import open
import pandas as pd

d1 = (sys.argv[1])
d2 = (sys.argv[2])
d3 = (sys.argv[3])
d4 = (sys.argv[4])
d5 = (sys.argv[5])
d6 = (sys.argv[6])

num = int(sys.argv[7])

file1 = open(d1, 'r', encoding="utf-8")
doc1 = file1.read().lower()
file1.close()

file2 = open(d2, 'r', encoding="utf-8")
doc2 = file2.read().lower()
file2.close()

file3 = open(d3, 'r', encoding="utf-8")
doc3 = file3.read().lower()
file3.close()

file4 = open(d4, 'r', encoding="utf-8")
doc4 = file4.read().lower()
file4.close()

file5 = open(d5, 'r', encoding="utf-8")
doc5 = file5.read().lower()
file5.close()

file6 = open(d6, 'r', encoding="utf-8")
doc6 = file6.read().lower()
file6.close()

# file1 = open(r'doc1.txt', 'r')
# doc1Sentences = file1.readlines()
# file1.close()
# file2 = open(r'doc2.txt', 'r')
# doc2Sentences = file2.readlines()
# file2.close()
# file3 = open(r'doc3.txt', 'r')
# doc3Sentences = file3.readlines()
# file3.close()
# file4 = open(r'doc4.txt', 'r')
# doc4Sentences = file4.readlines()
# file4.close()
# file5 = open(r'doc5.txt', 'r')
# doc5Sentences = file5.readlines()
# file5.close()
# file6 = open(r'doc6.txt', 'r')
# doc6Sentences = file6.readlines()
# file6.close()

finalDoc = (doc1+doc2+doc3+doc4+doc5+doc5+doc6)

words = word_tokenize(finalDoc)

stopWords = set(stopwords.words('english'))
stopRemoved = []
for w in words:
    if w in stopWords:
        continue
    stopRemoved.append(w)


wordDist = FreqDist(stopRemoved)
pos_tagged = nltk.pos_tag(stopRemoved)

NNtagsList = []
for word, tag in pos_tagged:
    if tag == 'NN':
        NNtagsList.append(word)

mostCommonWords = FreqDist(NNtagsList).most_common(20)
mostCommonWords = np.array(mostCommonWords)
df = pd.DataFrame(columns=('word', 'Document', 'Sentence'))

for i in mostCommonWords[0:num, 0]:
    for j in (sent_tokenize(doc1)):
        if i in (word_tokenize(j)):
            df = df.append({'word': i, 'Document': d1, 'Sentence': j}, ignore_index=True)
    #             print(i)
    #             print('doc1')
    #             print (j)
    for p in (sent_tokenize(doc2)):
        if i in (word_tokenize(p)):
            df = df.append({'word': i, 'Document': d2, 'Sentence': p}, ignore_index=True)
    #             print(i)
    #             print('doc2')
    #             print (p)
    for q in (sent_tokenize(doc3)):
        if i in (word_tokenize(q)):
            df = df.append({'word': i, 'Document':  d3, 'Sentence': q}, ignore_index=True)
    for r in (sent_tokenize(doc4)):
        if i in (word_tokenize(r)):
            df = df.append({'word': i, 'Document': d4, 'Sentence': r}, ignore_index=True)
    for s in (sent_tokenize(doc5)):
        if i in (word_tokenize(s)):
            df = df.append({'word': i, 'Document': d5, 'Sentence': s}, ignore_index=True)
    for t in (sent_tokenize(doc6)):
        if i in (word_tokenize(t)):
            df = df.append({'word': i, 'Document': d6, 'Sentence': t}, ignore_index=True)


dft = df.groupby(['word','Document'])['Sentence'].apply('\n'.join).reset_index()
dft2 = dft.groupby(['word']).agg({'Sentence':"\n".join, 'Document':','.join})

dft2.to_csv('out.csv')

print(dft2)
