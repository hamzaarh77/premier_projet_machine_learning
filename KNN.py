import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from math import dist
from collections import Counter
import random 
from scipy.stats import mode
from sklearn.model_selection import train_test_split


# on prend la classs majoritaire 
# on travaille sur les données d entrainement 
data=pd.read_csv("data/kanji_train_data.csv",sep=",",header=None)
data_c=pd.read_csv("data/kanji_train_target.csv",header=None)




# separartion des données en donnée d'entraienement et de validation 

X_train, X_valid, Y_train, Y_valid = train_test_split(
    data, data_c, test_size=0.2, random_state=90)



# on remets a jour les indices 
#les données d'entrainement et de validation sont bien allignés 
X_train.reset_index(inplace=True,drop= True)
Y_train.reset_index(inplace=True, drop=True)
X_valid.reset_index(inplace=True,drop= True)
Y_valid.reset_index(inplace=True, drop=True)


# on renormlise les données des ensemble d'entrainement 
mean=X_train.mean()
std=X_train.std() 
X_train=(X_train-mean)/std 

mean=X_valid.mean() 
std=X_valid.std() 
X_valid=(X_valid-mean)/std 

# on implemente l'algo du K plus proche voisin : KNN 
# fonction qui permet de calculer la distance euclidienne entre deux points 
def distance_euc(v1,v2):
    return np.linalg.norm(v1-v2,axis=1)
    
  

# fonction voisins qui envoie k plus proche voisins d'un point donnée
# cad trouve les voisins les plus proche de unb points et retourne les k premier 
# on a besoin de savoir leurs classes 


# obligé d'utilisé les numpy car parcourir le dataset avec une boucle prend trop de temps 
def voisins(point, X_train, k, Y_train):
    # transformation des parametres en numpy
    X_train_np = X_train.to_numpy()
    point_np = np.array(point)
    Y_train_ = Y_train.iloc[:, 0] 

    # distance euclidienne
    distances = distance_euc(X_train_np,point_np)


    # creation du df 
    df = pd.DataFrame({"label": Y_train_, "distance": distances})
    # trie by distance et retourne les k premiers 
    df_sorted = df.sort_values(by="distance").iloc[:k]

    return df_sorted



#print(voisins(X_valid.loc[0],X_train,5,Y_train))
    


def prediction(voisins):
    classes=list(voisins["label"])
    occ=Counter(classes)

    #print(classes)
    o=occ.most_common()[0][1]

    if len(occ)==len(classes): # cad tout les elements ont la mm occurence
        #print("choix rand")
        return random.choice(classes)
    else: 
        # dans le cas ou on a un seul element majorant on retourne sa class si on a plusieurs egaux on retourne un au hasard 
        elements_plus_commun=[item for item,count in occ.most_common() if count==o ]
        if len(elements_plus_commun)>1 :
            #print("choix rand entre elements max")
            return random.choice(elements_plus_commun)
        else :
            #print("choix plus commun")
            return occ.most_common(1)[0][0]

    



# generer un fichier csv ou on ecris nos predictions
test=pd.read_csv("data/kanji_test_data.csv",sep=",",header=None)
# normalizer le data set de test 
mean=test.mean() 
std=test.std()
test=(test-mean)/std



#predictions
predic=[]
for i in range(test.shape[0]):
    p=prediction(voisins(test.iloc[i,:],X_train,1,Y_train))
    predic.append(p)

#transforme la liste en numpy array 
predic_numpy=np.array(predic)
np.savetxt("sample.csv",predic_numpy)






# on evalue le modele en calculant le nombre de fois ou on a eu juste 
def evaluation(X_train,Y_train,X_valid,Y_valid,k):
    vrai=0
    total=len(X_valid)

    for i in range(X_valid.shape[0]):
        voisin_plus_proches=voisins(X_valid.iloc[i],X_train,k,Y_train)
        if prediction(voisin_plus_proches)==Y_valid.iloc[i,:][0] :
            vrai+=1
        print("score actuelle ----------->",str(vrai/total))


    return vrai/total

#print(evaluation(X_train,Y_train,X_valid,Y_valid,10))



# on trace une courbe qui represente la precision du modele en fonction de k 
# on trouve le meilleure k grace au donnée d"entrainement et de valdiation 
liste_score=[]

for k in range(1,5,2):
    liste_score.append(evaluation(X_train,Y_train,X_valid,Y_valid,k))

print(liste_score)

# la courbe :
liste_k=range(1,5,2)

plt.plot(liste_k,liste_score)
plt.xlabel("k")
plt.ylabel("score")

plt.show()


