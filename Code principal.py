import time
t = time.time()
import pandas as pd # only used to read the MNIST data set

train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)
#Separate features and targets
X = train_df.loc[:, train_df.columns != 187]
y = train_df.loc[:, train_df.columns == 187]

X_test = test_df.loc[:, test_df.columns != 187]
y_test = test_df.loc[:, test_df.columns == 187]
"""
    Implementation of the general ESN model.
"""

from easyesn.PredictionESN import PredictionESN

from easyesn import PredictionESN

ft=PredictionESN(187,300,1) #750 taille de reservoir


#print('Ok')

ft.setSpectralRadius(0.7)
#ft.optimize(X.head(200),y.head(200),X_test.head(200),y_test.head(200),1)
#ft.setFeedbackScaling(0.1)
ft.setLeakingRate(0.08)


from sklearn.utils import resample
df_1=train_df[train_df[187]==1]
df_2=train_df[train_df[187]==2]
df_3=train_df[train_df[187]==3]
df_4=train_df[train_df[187]==4]

df_0=(train_df[train_df[187]==0]).sample(n=2000,random_state=42)
df_1_upsample=resample(df_1,replace=True,n_samples=2000,random_state=123)
df_2_upsample=resample(df_2,replace=True,n_samples=2000,random_state=124)
df_3_upsample=resample(df_3,replace=True,n_samples=2000,random_state=125)
df_4_upsample=resample(df_4,replace=True,n_samples=2000,random_state=126)

#Il y a des beats répétés

train_df1=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
X1 = train_df1.loc[:, train_df1.columns != 187]
y1 = train_df1.loc[:, train_df1.columns == 187]
ft.fit(X1.head(len(X1)),y1.head(len(X1)),transientTime=0,
        transientTimeCalculationEpsilon=1e-3,
        transientTimeCalculationLength=20,
        verbose=0)

predictions=ft.predict(X_test)

#print('Ok')
import sklearn.metrics as sk

print(sk.r2_score(y_test, predictions ))
print(sk.mean_squared_error(y_test, predictions))


elapsed = time.time() - t
print(elapsed)
