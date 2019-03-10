import string
import pandas
from nltk.corpus import stopwords

from afinn import Afinn
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# list of 335 custom stopwords to account for context
context_stopwords_list = ["can","proposed","introduction","paper","propose","method","methods","show","approach", "algorithm", "algorithms",
"problem","problems","model","models","learning","data","also", "however","results","present","new","novel","using","two","based",
"demonstrate","used","different","first","based", "many","number","one","planning","agents","classification","existing","experiements",
"framework","function","information","performance","search","set","stateoftheart","structure","tasks","time","use","work","experiments",
"approaches","domain","efficient","given","important","knowledge","large","social","task","applications","complexity","constraints",
"domains","experimental","multiple","provide","presentation","significantly","space","study","users","well","agent","analysis",
"datasets","features","functions","games","general","graph","linear","optimal","optimization","order","previous","quality","real",
"realworld","presentation","several","solution","techniques","accuracy","better","class","computational","consider","cost",
"feature","improve","introduce","language","matrix","may","often","system","systems","target","training","representation","prediction",
"best","case","find","high","human","outperforms","particular","process","properties","selection","sets","soltuions","solve",
"solving","sparse","temporal","called","due","effectiveness", "evaluation","extensive","modeling","known","local","recent","result",
"solutions","state","strategy","constraint","design","develop","even","learn","machine","much","network","networks","objects","preferences",
"rules","simple","size","technique","terms","user","online","image","images","actions","variables","visual","various","value","distance",
"effective","input","instances","reasoning","probability","labels","latent","mapping","efficiency","efficiently","empirical","decision",
"allows","plan","prove","recently","recognition","setting","source","theoretical","three","thus","game","mechanism","address","complete",
"dynamic","objective","query","random","text","theory","able","automatically","dynamics","form","preference","provides","robust",
"significant","values","applied","finally","resulting","semantic","simultaneously","solvers","way","small","studies","world","action",
"among","approximation","bound","cases","compared","label","obtained","probabilistic","shown","standard","tensor","uses","via","becnhmark",
"classes","composition","examples","graphs","individual","logic","make","parameters","programs","propagation","settings","transfer",
"will","within","constant","dataset","object","reserach","similarity","video","without","challenging","consistency","content","current",
"distribution","finding","including","introduced","multiview","similar","single","utility","global","identity","sat","strategies",
"types","word","achieve","across","costs","describe","identity","influence","missing","possible","research","reward","accurate",
"analyze","equilibrium","evaluate","generated","identify","manifold","multiagent","stochastic","traditional","available","certain","factors",
"found","heuristic","heuristics","instance","investigate","popular","shows","therefore","translation","assumption","base","computing",
"effectively","either","empirically","exploration","generate","instead","loss","mehcanisms","people","support","underlying",
"generation","items","reduce","structured","achieves","web","issues","obtain","price","goal","main","making","since","specifically","example","need"]

# read csv file into panda dataframe
df = pandas.read_csv('FirstYear.csv')

# Afinn is a sentiment analysis library that assigns a specific value to each word in their premade bucket of words
# for example af.score("abandon") would give a score of -2 which would be negative since its less than 0
af = Afinn()

# this creates a list to put all of the abstracts in
list_of_abstracts = []

# this translator is later used to clean the words and take out string punctuation
# for example, 'hello!' would be changed to 'hello'
translator=str.maketrans('','',string.punctuation)

sentiment_categories = {}


# this computes all of the abstracts into our list(150 in this case)
for index in range(150):
    list_of_abstracts.append(df.loc[index, "Abstract"])

# this block of code will go through each abstract in our list and then after cleaning each word in the abstract use the
# AFinn library to tell us if the word is positive, negative or neutral and at the end sum all of the values to classify the abstract
for abstract in list_of_abstracts:
    sum = 0
    for word in abstract.split():
        # this strips the word of punctuation and converts it to lowercase
        temp = word.translate(translator).lower()
        # this checks to see if the word is a stopword, and if it is we continue to the next word
        if temp in stopwords.words() or temp in context_stopwords_list:
            continue

        sum += af.score(temp)
    # these three statements check to see if the abstract is classified as positive, negative or neutral
    if sum > 0:
        sentiment_categories['positive'] = sentiment_categories.get('positive', 0) + 1
    elif sum < 0:
        sentiment_categories['negative'] = sentiment_categories.get('negative', 0) + 1
    else:
        sentiment_categories['neutral'] = sentiment_categories.get('neutral', 0) + 1

print(sentiment_categories)

# this plots the results of the counts of each category of sentiment
objects = ('Positive', 'Negative', 'Neutral')
y_pos = np.arange(len(objects))
counts = [sentiment_categories['positive'], sentiment_categories['negative'], sentiment_categories['neutral']]

plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Count')
plt.title('Sentiment Analysis Based off Abstract (Year 1 Excluding Context Stopwords)')

plt.show()
