import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# get dataset; names are strings and genders are listed as 0 for female, 1 for male
df = pd.read_csv('2017_top_baby_names.csv')
# df = pd.read_csv('names_genders.csv')

# break the names up into parts
def features(name):
    return f"{name[-1]} {name[-2:]} {name[-3:]}"

# extract features from names
Xfeatures = np.array(df['Name'])
for i in range(len(Xfeatures)):
    Xfeatures[i] = features(Xfeatures[i])
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

# label is gender, already numerical
y = np.array(df['Gender'])

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# use the Naive Bayes classifier for multinomial models
clf = MultinomialNB()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print('Accuracy:', accuracy)

# let the user enter a name
while True:
    user_name = str(input('Enter a name: '))
    user_name_data = np.array([user_name])
    user_name_data[0] = features(user_name_data[0])
    user_name_data_vector = cv.transform(user_name_data)
    guess = clf.predict(user_name_data_vector)
    if guess[0] == 0:
        print('I think that name is female.')
    elif guess[0] == 1:
        print('I think that name is male.')
    else:
        print(guess)

# test some sample names
# sample_names = [] # put some names into this list
# sample_test = np.array(sample_names)
# for i in range(len(sample_test)):
#     sample_test[i] = features(sample_test[i])
# test_data = cv.transform(sample_test)
# guess = clf.predict(test_data)
#
# for i in range(len(sample_names)):
#     print(sample_names[i], guess[i])
