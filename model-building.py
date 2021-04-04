import pandas as pd
transfusion_data = pd.read_csv('transfusion/transfusion.data')
df = transfusion_data.copy()
target = 'whether he/she donated blood in March 2007'
# Separating X and y
X = df.drop('whether he/she donated blood in March 2007', axis=1)
Y = df['whether he/she donated blood in March 2007']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('transfusion_clf.pkl', 'wb'))
