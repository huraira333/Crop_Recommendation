
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle



df=pd.read_csv("F:\Ids project\Ids project\data.csv")
df.head()

### divide into labels and features ###
x=df.drop(['label'],axis=1)
y=df['label']

### dividing data into training-testing splits

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


### Logistic Regression ###
#Import the Model
model=LogisticRegression()
### Model Training
model.fit(X_train,y_train)

# Saving model to disk
pickle.dump(model, open('model_fertilizer.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model_fertilizer.pkl','rb'))

print(model.predict([[90, 40, 40,20,80,7,200]]))




