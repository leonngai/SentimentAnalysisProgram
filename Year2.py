import string
import pandas
from nltk.corpus import stopwords

from afinn import Afinn
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# read csv file into panda dataframe
df2 = pandas.read_csv('SecondYear.csv')

# Afinn is a sentiment analysis library that assigns a specific value to each word in their premade bucket of words
# for example af.score("abandon") would give a score of -2 which would be negative since its less than 0
af = Afinn()

# this creates a list to put all of the abstracts in
list_of_abstracts = []

# this translator is later used to clean the words and take out string punctuation
# for example, 'hello!' would be changed to 'hello'
translator=str.maketrans('','',string.punctuation)

sentiment_categories = {}

# this computes all of the abstracts into our list
for index in range(397):
    list_of_abstracts.append(df2.loc[index, "abstract"])

# this block of code will go through each abstract in our list and then after cleaning each word in the abstract use the
# AFinn library to tell us if the word is positive, negative or neutral and at the end sum all of the values to classify the abstract
for abstract in list_of_abstracts:
    sum = 0
    for word in abstract.split():
        # this strips the word of punctuation and converts it to lowercase
        temp = word.translate(translator).lower()
        # this checks to see if the word is a stopword, and if it is we continue to the next word
        if temp in stopwords.words():
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
plt.title('Sentiment Analysis Based off Abstract (Year 2)')

plt.show()
