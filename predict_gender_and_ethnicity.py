
# coding: utf-8

# # Gender Usernames/Names

# In[1]:


import numpy as np
import pandas as pd
#from tqdm import tqdm
#tqdm.pandas()
np.random.seed(7)

import tensorflow as tf
load_model = tf.keras.models.load_model
adam = tf.keras.optimizers.Adam(lr=1e-3)

import string
alphabet_list = list(string.ascii_lowercase)
max_name_len = 20

import re

def string_vectorizer(strng, alphabet, max_str_len=20, gender=True):
    if(gender):
        strng = re.sub(r"[^a-z]+", "", strng.lower())
    else:
        strng = re.sub(r"[^a-zA-z0-9-]+", "", strng)
    vector = [[0 if char != letter else 1 for char in alphabet] for letter in strng[0:max_str_len]]
    while len(vector) != max_str_len:
        vector = [*vector, [0 for char in alphabet]]
    return np.array(vector)


gendermodel = load_model("gendermodel.h5")
gendermodel.load_weights("genderweights.h5")
gendermodel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# 90% accuracy on 30k female and 33k male names synthesised from Indian names list
gendermodel.summary()


# In[2]:


names = pd.Series(["RajSood25", "mukta57cute", "burnvipulkumarfire", "priyasubramanium"])
print(names)
names_transform = names.apply(lambda name: string_vectorizer(name, alphabet_list, max_name_len).reshape(1, 20, 26))
names_transform = np.vstack(names_transform.tolist())
prediction = gendermodel.predict(names_transform)
print("array([[male, female]]) probability")
prediction = [[int(pred[0]*100)/100, int(pred[1]*100)/100] for pred in prediction]
print(np.array(prediction))


# # Ethnicity Usernames

# In[4]:



alphabet_listset = pd.read_csv("ethnicity_listset.csv")["characters"].tolist()
print(alphabet_listset[0:5])
ethnicitymodel = load_model("eth_model.h5")

ethnicitymodel.load_weights("eth_weights.h5")
ethnicitymodel.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# 88% accuracy on Indian vs non indian usernames
ethnicitymodel.summary()


# In[5]:


names = pd.Series(["RajkumarSood", "mukta57", "johnsmith", "KeanuReeves", "Elias", "priyasubramanium", "Devashish"])
print(names)
names_transform = names.apply(lambda name: string_vectorizer(name, alphabet_listset, max_name_len, False).reshape(1, 20, 63))
names_transform = np.vstack(names_transform.tolist())
prediction = ethnicitymodel.predict(names_transform)
print("array([[other, indian ethnicity]]) probability")
prediction = [[int(pred[0]*100)/100, int(pred[1]*100)/100] for pred in prediction]
print(np.array(prediction))


# In[ ]:


# Star this repo to add to your favorite repos or fork the code!

