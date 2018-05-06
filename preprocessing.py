
# coding: utf-8

# In[264]:


import numpy as np
import pandas as pd


# # Pre-processing 2015-16 Annual Files with Cause-of-Death
# Retain all fields pertaining to:
# * **State file number** (i.e., certificate number)
# * **Underlying Cause-of-Death** (single cause)
# * **Multiple Cause-of-Death** (up to 20 causes)
# 
# Add **year** field to designate between 2015 and 2016 records.
# 
# Drop all other fields
# 
# Deaths with underlying or multiple cause of death for the following ICD codes will be flagged as a DMI death: X40–X44, X60–X64, X85, or Y10–Y14

# In[2]:


deaths2015 = pd.read_excel('2015-deaths-annual-file.csv')
deaths2015.head()


# In[3]:


deaths2016 = pd.read_excel('2016-deaths-annual-file.csv')
deaths2016.head()


# In[4]:


deaths2015.columns.values


# In[88]:


deaths2015['year'] = 2015
deaths2015.rename(index=str, columns={'certno': 'State file number', 
                                        'underly3': 'Underlying COD code 3'}, inplace=True)
deaths2015[['State file number','Underlying COD code 3','mltcse1','mltcse2','mltcse3','mltcse4','mltcse5','mltcse6','mltcse7','mltcse8','mltcse9','mltcse10','mltcse11','mltcse12','mltcse13','mltcse14','mltcse15','mltcse16','mltcse17','mltcse18','mltcse19','mltcse20','year']].head()


# In[6]:


deaths2016.columns.values


# In[91]:


deaths2016['year'] = 2016
deaths2016.rename(index=str, columns={'certno': 'State file number', 
                                     'underly3': 'Underlying COD code 3',
                                     'Record Axis Code 1': 'mltcse1',
                                     'Record Axis Code 2': 'mltcse2',
                                     'Record Axis Code 3': 'mltcse3',
                                     'Record Axis Code 4': 'mltcse4',
                                     'Record Axis Code 5': 'mltcse5',
                                     'Record Axis Code 6': 'mltcse6',
                                     'Record Axis Code 7': 'mltcse7',
                                     'Record Axis Code 8': 'mltcse8',
                                     'Record Axis Code 9': 'mltcse9',
                                     'Record Axis Code 10': 'mltcse10',
                                     'Record Axis Code 11': 'mltcse11',
                                     'Record Axis Code 12': 'mltcse12',
                                     'Record Axis Code 13': 'mltcse13',
                                     'Record Axis Code 14': 'mltcse14',
                                     'Record Axis Code 15': 'mltcse15',
                                     'Record Axis Code 16': 'mltcse16',
                                     'Record Axis Code 17': 'mltcse17',
                                     'Record Axis Code 18': 'mltcse18',
                                     'Record Axis Code 19': 'mltcse19',
                                     'Record Axis Code 20': 'mltcse20'}, inplace=True)
deaths2016[['State file number','Underlying COD code 3','mltcse1','mltcse2','mltcse3','mltcse4','mltcse5','mltcse6','mltcse7','mltcse8','mltcse9','mltcse10','mltcse11','mltcse12','mltcse13','mltcse14','mltcse15','mltcse16','mltcse17','mltcse18','mltcse19','mltcse20','year']].head()


# In[147]:


def flagDMI(row):
    """Check 3 digit ICD code for underlying and multiple causes of death
       Return: 1 if code is in ICD code subset, 0 otherwise
    """
    
    for column in row:
        if column[:3] in ['X40','X41','X42','X43','X44','X60','X61','X62','X63','X64','X85','Y10','Y11','Y12','Y13','Y14']:
            return 1
    return 0


# In[148]:


# Concatenate 2015 and 2016 annual files
deaths = pd.concat([deaths2015[['State file number','Underlying COD code 3','mltcse1','mltcse2','mltcse3','mltcse4','mltcse5','mltcse6','mltcse7','mltcse8','mltcse9','mltcse10','mltcse11','mltcse12','mltcse13','mltcse14','mltcse15','mltcse16','mltcse17','mltcse18','mltcse19','mltcse20','year']],
                    deaths2016[['State file number','Underlying COD code 3','mltcse1','mltcse2','mltcse3','mltcse4','mltcse5','mltcse6','mltcse7','mltcse8','mltcse9','mltcse10','mltcse11','mltcse12','mltcse13','mltcse14','mltcse15','mltcse16','mltcse17','mltcse18','mltcse19','mltcse20','year']]], join="outer")

# Replace NaN values with *
deaths.fillna('*', inplace=True)

# Flag DMI as 1 if in ICD code subset and 0 otherwise
deaths['DMI'] = deaths[['Underlying COD code 3','mltcse1','mltcse2','mltcse3','mltcse4','mltcse5','mltcse6','mltcse7','mltcse8','mltcse9','mltcse10','mltcse11','mltcse12','mltcse13','mltcse14','mltcse15','mltcse16','mltcse17','mltcse18','mltcse19','mltcse20']].apply(flagDMI, axis=1)    

# Drop underlying and multiple causes of death
deaths = deaths[['State file number','year','DMI']]

deaths.head()


# In[146]:


# Count number of DMI versus non-DMI deaths
deaths.groupby('DMI')['DMI'].count()


# # Pre-processing 2015-16 Literal Text Files
# Retain all fields pertaining to:
# * **State file number** (i.e., certificate number)
# * The chain of events leading to death (from Part I)
#   * **Cause-of-Death Line A**
#   * **Cause-of-Death Line B**
#   * **Cause-of-Death Line C**
#   * **Cause-of-Death Line D**  
# * Other siginificant conditions that contributed to cause of death (from Part II)
#   * **Other Significant Conditions**
# * How the injury occurred (in the case of deaths due to injuries [from Box 43])
#   * **How the Injury Occurred**   
#   
# Add **year** field to designate between 2015 and 2016 records.
# 
# Drop fields pertaining to:
# * **Interval Time - Line (A/B/C/D)**
# * **COD-DUE-TO-(B/C/D)** - Further inspection shows only one record in 2015 using this field
# * **Injury Place**

# In[51]:


# Load 2015 and 2016 literal text files
literals2016 = pd.read_csv('2016-deaths-literal-text.csv', encoding='latin1')
literals2015 = pd.read_csv('2015-deaths-literal-text.csv', encoding='latin1')


# In[52]:


# Examine field names for 2015 literals file
literals2015.columns.values


# In[56]:


# Examine field names for 2016 literals file
literals2016.columns.values


# In[159]:


# Add year field and rename fields to conform to 2016 labels
literals2015['year'] = 2015
literals2015.rename(index=str, columns={'CERT-NUM': 'State file number', 
                                        'COD-TEXT-1': 'Cause of Death - Line A', 
                                        'COD-TEXT-2': 'Cause of Death - Line B', 
                                        'COD-TEXT-3': 'Cause of Death - Line C', 
                                        'COD-TEXT-4': 'Cause of Death - Line D',
                                        'COD-INTERVAL-1': 'Interval Time - Line A', 
                                        'COD-INTERVAL-2': 'Interval Time - Line B', 
                                        'COD-INTERVAL-3': 'Interval Time - Line C',
                                        'COD-INTERVAL-4': 'Interval Time - Line D', 
                                        'COD-DUE-TO-B': 'COD-DUE-TO-B', 
                                        'COD-DUE-TO-C': 'COD-DUE-TO-C', 
                                        'COD-DUE-TO-D': 'COD-DUE-TO-D',
                                        'COD-OTHER-TEXT': 'Other Significant Conditions', 
                                        'INJURY-DESC': 'How Injury Occurred', 
                                        'INJURY-PLACE': 'Place of Injury'}, inplace=True)
literals2015.head()


# In[158]:


# Add year field
literals2016['year'] = 2016
literals2016.rename(index=str, columns={'State File Number': 'State file number'}, inplace=True)
literals2016.head()


# In[57]:


# Examine COD-DUE-TO fields to determine if there is information of value
literals2015[['COD-DUE-TO-B','COD-DUE-TO-C','COD-DUE-TO-D']].dropna(how='all')


# In[160]:


# Concatenate two 2015 and 2016 literals files and drop unneccessary fields
literals = pd.concat([literals2015,literals2016], join="outer")
literals.drop(['COD-DUE-TO-B','COD-DUE-TO-C','COD-DUE-TO-D','Interval Time - Line A','Interval Time - Line B','Interval Time - Line C','Interval Time - Line D','Place of Injury'], axis=1, inplace=True)
literals


# # Join Annual and Literal Text Files

# In[156]:


deaths.head()


# In[161]:


literals.head()


# In[171]:


# Perform left join of annual and literal text datasets
combined = deaths.merge(right=literals, on=['State file number','year'], how='left')
combined


# In[176]:


# Fill missing literal text values with spaces and combine into a single text field
combined.fillna(' ', inplace=True)
combined['literal text'] = combined.apply(lambda x:'%s %s %s %s %s %s' % (x['Cause of Death - Line A'], x['Cause of Death - Line B'], x['Cause of Death - Line C'], x['Cause of Death - Line D'], x['How Injury Occurred'], x['Other Significant Conditions']),axis=1)


# In[226]:


# Drop all fields except for the DMI targets and combined literal text field
literal_text = combined.filter(items=['DMI','literal text'])
literal_text


# In[227]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Run first time to download NLTK packages
# nltk.download()


# In[228]:


# Create list of stop words
stop_words = set(map((lambda x: x.upper()),stopwords.words('english')))
stop_words


# In[354]:


def cleanText(literal_text):
    """Given a string of text, remove punctuation, stop words and numbers"""
    words = RegexpTokenizer(r'\w+').tokenize(literal_text)
    words_stripped  = [word for word in words if word not in stop_words and str.isalpha(word[0])]
    return ' '.join(words_stripped)


# In[355]:


# Remove punctuation and stop words from literal text
literal_text['literal text'] = literal_text['literal text'].apply(cleanText)

# Remove rows with blank literal text
literal_text = literal_text[literal_text['literal text'] != ""]

literal_text


# In[356]:


# Export literal text data frame to CSV file. TF matrix too large to export (800 MB+)
literal_text.to_csv(path_or_buf='literal-text.csv',sep=',',)


# # Create Term-Frequency Matrix

# In[234]:


from sklearn.feature_extraction.text import CountVectorizer


# In[357]:


# Create term-frequency matrix:
#   Original dataset sets all words to uppercase, leave as is
#   Frequencies are binary (0 or 1)
#   Only words that appear a minimum of 5 times in the dataset are counted
cv = CountVectorizer(lowercase=False, binary=True, min_df=5)
tf = cv.fit_transform(literal_text['literal text'])


# In[358]:


# Export bag of words
with open('bag-of-words.txt','w') as file:
    file.write("\n".join(cv.get_feature_names()))
    
print("Number of words:",len(cv.get_feature_names()))


# In[359]:


# Create as pandas dataframe with DMI targets
tf_df = pd.DataFrame(tf.A, columns=cv.get_feature_names())
tf_df['DMI'] = literal_text['DMI']

# Remove any rows with DMI missing
tf_df.dropna(subset=['DMI'], inplace=True)


# In[360]:


tf_df.head()


# In[382]:


# Pickle fitted CountVectorizer model and numpy array of term frequencies

import pickle

with open('count_vectorizer_model.pkl', 'wb') as file:

    # Pickle CountVectorizer model
    pickle.dump(cv, file)
    
with open('term-frequencies.pkl', 'wb') as file:

    # Pickle CountVectorizer model
    pickle.dump(tf, file)    


# In[376]:


# Test pickled CountVectorizer model

with open('count_vectorizer_model.pkl', 'rb') as file:

    # Pickle CountVectorizer model
    load_cv = pickle.load(file)

load_cv.get_feature_names()



# In[375]:


# Test pickled numpy array of term-frequencies

with open('term-frequencies.pkl', 'rb') as file:

    # Pickle CountVectorizer model
    load_tf = pickle.load(file)

load_tf.A

