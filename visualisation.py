import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# on affiche les kanji presente dans kanji_train_data.csv 
data=pd.read_csv("data/kanji_train_data.csv",sep=",",header=None)

for i in range(data.shape[0]):
    vect=data.iloc[i,:].to_numpy()
    matrix=vect.reshape((64,64))

    # affichage de l'image 
    plt.imshow(matrix,cmap='gray')
    plt.axis('off')
    plt.show()