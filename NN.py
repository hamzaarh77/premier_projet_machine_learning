import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch as th
import torch.nn as nn
import torch.optim as optim

# Chargement et préparation des données
data = pd.read_csv("data/kanji_train_data.csv", sep=",", header=None)
data_c = pd.read_csv("data/kanji_train_target.csv", header=None)
test=pd.read_csv("data/kanji_test_data.csv",header=None)

# Séparation des données en ensembles d'entraînement et de validation
X_train, X_valid, Y_train, Y_valid = train_test_split(
    data, data_c, test_size=0.2, random_state=90)

# Normalisation des données
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std

# normalizer les données de test 
mean=test.mean()
std=test.std() 
test=(test - mean) / std

# pas besoin d'ajouter l'interpecte 
#conversion en tenseur 
X_t = th.tensor(X_train.values, dtype=th.float32)
Y_t = th.tensor(Y_train.values.squeeze(), dtype=th.long) 
X_v = th.tensor(X_valid.values, dtype=th.float32)
Y_v = th.tensor(Y_valid.values.squeeze(), dtype=th.long)
t_t = th.tensor(test.values, dtype=th.float32)

# modele :
class Reg(nn.Module):
    def __init__(self, nb_colones, nb_classes):
        super(Reg, self).__init__()
        self.linear = nn.Linear(nb_colones, nb_classes)
    
    def forward(self, x):
        x= self.linear(x)
        return x ;



nb_colones = X_t.shape[1]
nb_classes = 20
eta=0.01

# modele :
model = Reg(nb_colones, nb_classes)


# fonctions :
entropy = nn.CrossEntropyLoss()
gradient = optim.SGD(model.parameters(), lr=eta)

# Entraînement du modèle
num_iterations = 1000
for i in range(num_iterations):
    gradient.zero_grad() # remets a zero le gradient (a cause de l'iteration precedente)
    outputs = model(X_t)
    perte = entropy(outputs, Y_t)
    perte.backward() # calcule le gradient de la fct de perte 
    gradient.step() # mets a jour les poids 

 

# evaluation
with th.no_grad():
    logits = model(t_t)
    predictions = th.argmax(logits,dim=1)
    # transformer en numpy 
    predictions_numpy=predictions.numpy()
    np.savetxt("RL.csv",predictions)







