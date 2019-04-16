convertinrki#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:36:33 2018

@author: samas
"""

#Simple Text Processing
import re 
data="Hello Everyone My Mobile number is 9999887666 what is yours?"
print(re.sub('\D','',data))

print(re.sub('\d','*',data))


data='''Hi Everyone hope u r doing good, you can connect me at abc@xyz.com,
you can also connect at anytime, the support mail is support_helpline@gmail.com
please also keep boss_12@gmail.com in cc'''
pattern='[a-zA-Z0-9._]+@[a-zA-Z0-9.]+'
print(re.findall(pattern,data))

#Speach To Text Conversion

import speech_recognition as sr

r=sr.Recognizer()
with sr.Microphone(1) as source:
    audio=r.listen(source,phrase_time_limit=10)
    
    
text=r.recognize_google(audio)
print(text)"""

#######   Tokenization    #######

data='''Bangalore is capital of karnataka state in southern india.I live in
Bangalore. I have so many friends here. we are visiting Mysore next Month.
Mr John is my friend who work in a Software company.his email is ashjh@gmail.com.'''
print('####### Tokenizer ######')
print(data.split('.'))

from nltk import sent_tokenize
print(sent_tokenize(data))

import nltk
nltk.download('punkt')

from nltk import word_tokenize
print(word_tokenize(data))

#### Morphological Analysis ####

#Streaming
from nltk.stem import PorterStemmer
ps=PorterStemmer()

ps.stem('cars')  #Out car

ps.stem('boxes') #Out box


ps.stem('wives') #Out wive

#lematization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wd=WordNetLemmatizer()

wd.lemmatize('children') #Out child

wd.lemmatize('wives') #Out wife

from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
pos_tag(['going','Delhi','Laptop','is'])

nltk.download('tagsets')
nltk.help.upenn_tagset('VBZ')

data=[('The food was delicious, I love the food','classA'),
    ('The iceceeam made the day , it was awesome','classA'),
    ('The Burger was stale it was bad','classB'),
    ('my boss is a monster and I hate him','classB'),
    ('The restaurant was untidy and i have such place','classB'),
    ('Pizza was bad and I regret paying for it','classB'),
    ('My wife love pizza more than me','classA'),
     ('the soup was awesome and tasty','classA'),
     ('I just love these momos, it was just amazing','classA'),
     ('the delivery made the food worst I hate this service','classB'),
     ('Do not serve the stale food its too bad','classB'),
     ('Maggie is always awesome','classA')]


x=[]
y=[]
for i in data:
    x.append(i[0])
    y.append(i[1])


from sklearn.feature_extraction.text import CountVectorizer
cvec=CountVectorizer(lowercase=True,stop_words='english')
xd=cvec.fit_transform(x).toarray()

xd.shape

cvec.get_feature_names()


### TEXTBLOB  ####
from textblob import TextBlob

data=TextBlob('Hello Everyone How r u')

data.translate(to='ka') #TextBlob("გაუმარჯოს ყველას")

data=TextBlob('Race 3 is a worst movie')
data.sentiment.polarity   # -1.0

data=TextBlob('I havv lost my watch.')
data.correct()  # TextBlob("I have lost my watch.")

data=TextBlob('havv')
data.correct() 

####  ChatBot Platforms  ####

import chatterbot
bot=chatterbot.ChatBot('bot',trainer='chatterbot.trainer_ChatterBotCorpusTrainer')
bot.train('chatter.corpus.english')

while True:
    qus=input('You: ')
    if qus=='end':
        break
    ans=bot.get_response(qus)
    print("bot:",ans)
        
    
    
    
    


















