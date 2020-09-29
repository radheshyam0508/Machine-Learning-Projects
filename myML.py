import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle



# Splitind data

def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size= int(len(data)*ratio)
    test_indices= shuffled[:test_set_size]
    train_indices= shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__=="__main__": 

    # Read The Data

    df=pd.read_csv('data.csv')
    #df.head()
    #df.info()
    train,test=data_split(df,0.2)
    # convert in to numpy array
    x_train=train[['bodyTemp','bodyPain','runnyNose','diffBreath','o2Saturation','travelHistory','age','LossofTasteSmell','vomiting','Diarrhea']].to_numpy()
    x_test=test[['bodyTemp','bodyPain','runnyNose','diffBreath','o2Saturation','travelHistory','age','LossofTasteSmell','vomiting','Diarrhea']].to_numpy()
    y_train=train[['infectionProb']].to_numpy()
    y_test=test[['infectionProb']].to_numpy()
    # Reshape The data
    y_train=train[['infectionProb']].to_numpy().reshape(2000,)
    y_test=test[['infectionProb']].to_numpy().reshape(499,)


    # LOgisticRegression

    clf=LogisticRegression()
    clf.fit(x_train,y_train)

    #clf.predict([[99.82,1,0,-1,89,1,36,0,1,0]])

    

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()

     # Code for inference
    #inputFeatures= ([[99.82,1,0,1,89,1,36,0,1,0]])
    #infProb=clf.predict_proba(inputFeatures)[0][1]

   