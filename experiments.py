
# coding: utf-8

# # CS737 Final Project
# Author: Anthony Lipphardt
# 
# Date: April 23, 2018

# In[9]:


import pandas as pd
import numpy as np
from time import time


# In[108]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


# ## Import and Balance Dataset

# In[2]:


literal_text = pd.read_csv('literal-text.csv', index_col=0)
literal_text.head()


# In[24]:


# Create random undersampling of over represented class and create final balanced dataset

undersample = literal_text[literal_text['DMI'] == 0].sample(n=len(literal_text[literal_text['DMI'] == 1]))
DMIrows = literal_text[literal_text['DMI'] == 1]

final = pd.concat([undersample, DMIrows])
final.groupby('DMI').count()


# # Find Optimal Values for LSA
# Using optimal parameters for CountVectorizer, find values for n_components hyperparameter that retains 85, 90, and 95 percent of explained variance in the dataset.

# In[50]:


# Convert data using CountVectorizer with optimal parameters and run dimensionality reduction

cv = CountVectorizer(lowercase=False, binary=True, min_df=3, ngram_range=(1, 2))
tf = cv.fit_transform(final['literal text'])
print("Number of words:",len(cv.get_feature_names()))


# In[72]:


# Find values for n_components that find 85%, 90%, and 95% variance

for i in (525, 775, 1180):
    svd = TruncatedSVD(n_components=i)
    svd.fit(tf)
    print(np.sum(svd.explained_variance_ratio_))


# ## Setup Train and Test Data
# Split final balanced dataset into target and test sets using a 75:25 split. Data and targets will be separated.

# In[93]:


# Split data into training and testing
train, test, traint, testt = train_test_split(final['literal text'], final['DMI'], test_size=0.25)


# ## Configure Pipelines and Parameters for Grid Search

# In[88]:


# Create pipelines for Naive Bayes and SVM workflows

NB_pipeline = Pipeline([
    ('NBvect', CountVectorizer(lowercase=False,binary=True)),
    ('NBclf', BernoulliNB(binarize=None))
])

SVM1_pipeline = Pipeline([
    ('SVMvect', CountVectorizer(lowercase=False,binary=True)),
    ('SVMclf', SVC(kernel='linear'))
])

SVM2_pipeline = Pipeline([
    ('SVMvect', CountVectorizer(lowercase=False,binary=True)),
    ('SVMdim', TruncatedSVD()),
    ('SVMclf', SVC(kernel='linear'))
])


# In[135]:


# Create parameter grids for Naive Bayes and SVM workflows

NB_parameters = {
    
    'NBvect__min_df': (3,5),
    'NBvect__ngram_range': ((1,1),(1,2)),
    
    'NBclf__alpha': (0, 0.1, 0.5, 1)
    
}


SVM1_parameters = {
    
    'SVMvect__min_df': (3,5),
    'SVMvect__ngram_range': ((1,1),(1,2)),
  
    'SVMclf__C': (1, 10, 100, 1000)
    
}

SVM2_parameters = {
    
    'SVMvect__min_df': (3,),    
    'SVMvect__ngram_range': ((1,2),),    

    'SVMdim__n_components': (525, 775, 1180),

    'SVMclf__C': (1,)
    
}


# In[126]:


def runTests(data, targets, pipeline, parameters):

    """ Perform grid search with specified pipeline and parameters
        on data training set with targets as labels
        
        Evaluate performance based on precision and print parameters
        for best estimator
        
        grid search object is returned for further analysis"""

    grid_search = GridSearchCV(pipeline, parameters, verbose=1, cv=10, scoring='precision')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(data, targets)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    return grid_search


# ## Run Grid Search for Each Pipeline

# In[136]:


# Run grid search for Naive Bayes
NB_grid_search = runTests(train, traint, NB_pipeline, NB_parameters)


# In[128]:


# Run grid search for SVM without dimensionality reduction
SVM1_grid_search = runTests(train, traint, SVM1_pipeline, SVM1_parameters)


# In[131]:


# Run grid search for SVM with dimensionality reduction
SVM2_grid_search = runTests(train, traint, SVM2_pipeline, SVM2_parameters)


# # Examine and Export Grid Search Results

# In[137]:


# Gather results from grid search
NB_results = pd.DataFrame({'params': NB_grid_search.cv_results_['params'], 'Classifier': 'Naive Bayes', 'mean_test_score': NB_grid_search.cv_results_['mean_test_score']})
SVM1_results = pd.DataFrame({'params': SVM1_grid_search.cv_results_['params'], 'Classifier': 'SVM w/o Reduction', 'mean_test_score': SVM1_grid_search.cv_results_['mean_test_score']}) 
SVM2_results = pd.DataFrame({'params': SVM2_grid_search.cv_results_['params'], 'Classifier': 'SVM with Reduction', 'mean_test_score': SVM2_grid_search.cv_results_['mean_test_score']})
grid_search_results = pd.concat([NB_results, SVM1_results, SVM2_results], ignore_index=True)
grid_search_results


# In[146]:


print("Naive Bayes")
print("  Average fit time:",np.mean(NB_grid_search.cv_results_['mean_fit_time']))
print("  Average score time:",np.mean(NB_grid_search.cv_results_['mean_score_time']))


# In[147]:


print("Support Vector Machine (w/o LSA)")
print("  Average fit time:",np.mean(SVM1_grid_search.cv_results_['mean_fit_time']))
print("  Average score time:",np.mean(SVM1_grid_search.cv_results_['mean_score_time']))


# In[148]:


print("Support Vector Machine (w/ LSA)")
print("  Average fit time:",np.mean(SVM2_grid_search.cv_results_['mean_fit_time']))
print("  Average score time:",np.mean(SVM2_grid_search.cv_results_['mean_score_time']))


# In[139]:


# Export grid search results to CSV
grid_search_results.to_csv(path_or_buf='grid-search-results.csv',sep=',')


# ## Run Optimal Classifier Against Test Data
# Fit training data to optimal classifier, transform test data, and obtain predictions.
# 
# Classification report and confusion matrix will be computed. Focus is on scoring for precision and specificity of drug mention with involvement (DMI) death.

# In[166]:


# Fit to target data using optimal parameters in grid search and run on test data
cv = CountVectorizer(lowercase=False,binary=True, min_df=3, ngram_range=(1,2))
tf = cv.fit_transform(train)

svd = TruncatedSVD(n_components=1180)
tf_svd = svd.fit_transform(tf)

clf = SVC(kernel='linear', C=1)
clf.fit(tf_svd, traint)

predicted = clf.predict(svd.transform(cv.transform(test)))

precision = precision_score(testt, predicted, average=None)

print(classification_report(testt, predicted, target_names=['Non-DMI','DMI']))

print("\nConfusion Matrix:")
print(confusion_matrix(testt, predicted))


# ## Examine Misclassified Records

# In[164]:


print("RECORDS MISCLASSIFIED AS DMI")
print("====================================\n")
for record in test[(predicted == 1) & (testt == 0)]:
    print(record,"\n")


# In[163]:


print("RECORDS MISCLASSIFIED AS NON-DMI")
print("====================================\n")
for record in test[(predicted == 0) & (testt == 1)]:
    print(record,"\n")

