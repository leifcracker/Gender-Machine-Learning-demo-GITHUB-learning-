from sklearn import neural_network

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

### data set above is atypical of avg human population
### will changing classifier impact ability to learn
### cl1 - DecisionTreeClassifier
### cl2 - RandomForestClassifier
### cl3 - MLPClassifier
clf = neural_network.MLPClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[240,93,45]])

print(prediction)
