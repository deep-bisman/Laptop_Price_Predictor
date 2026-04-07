import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import pickle 
df1=pd.read_csv('cleaned_laptop.csv')
print(df1.head(5))
#print(df1.shape)

x=df1.drop(columns=['Price'])
#print(x)
y=np.log(df1['Price'])
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=2)
step1=ColumnTransformer(transformers=[('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])],remainder='passthrough')

step2=RandomForestRegressor(n_estimators=100,random_state=3,max_samples=0.5,max_features=0.75,max_depth=15)

pipe=Pipeline([('step1',step1),('step2',step2)])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

#print('r2 score',r2_score(y_test,y_pred))
#print('mean abs. error',mean_absolute_error(y_test,y_pred))

# here we checked all algo and best result was given by randomforest


pickle.dump(df1,open('df1.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))