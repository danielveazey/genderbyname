import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# get dataset; names are strings and genders are listed as 0 for female, 1 for male
df = pd.read_csv('names_genders.csv')

# extract features from names
Xfeatures = np.array(df['Name'])
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

# label is gender, already numerical
y = np.array(df['Gender'])

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# use the Naive Bayes classifier for multinomial models
clf = MultinomialNB()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)

print('Accuracy:', accuracy)
