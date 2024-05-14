

Ce projet comprend le développement et l'évaluation de divers modèles d'apprentissage automatique, y compris la visualisation d'images à partir de données, le K-Plus Proches Voisins (KNN), la Régression Logistique Multivariée et les Réseaux de Neurones. Chaque modèle est appliqué à des ensembles de données spécifiques dans le but de classification multivariées.







## Visualisation d'Images à partir de Données Tabulaires

L'objectif est de charger des données d'image stockées sous forme tabulaire à partir d'un fichier CSV et de les visualiser sous forme de matrice d'images.

### Étapes :

- **Chargement des Données :** Utilisation de pandas pour charger les données d'image à partir d'un fichier CSV. Chaque ligne représente une image aplatie.
- **Algorithme :** Transformation de chaque vecteur d'image aplatie en une matrice 64x64 et affichage à l'aide de matplotlib avec une carte de couleurs grise.








## K-Plus Proches Voisins (KNN)

L'algorithme KNN est employé pour la classification basée sur les exemples d'entraînement les plus proches dans l'espace des caractéristiques.

### Étapes :

- **Préparation des Données :** Division des données en ensembles d'entraînement et de validation en utilisant `train_test_split` et normalisation des données.
- **Calcul de la Distance Euclidienne :** Implémentation d'une fonction pour calculer la distance euclidienne entre deux vecteurs.
- **Trouver les Plus Proches Voisins :** Trier les points d'entraînement par leur distance par rapport à un point donné et sélectionner les `k` premiers.
- **Prédiction de la Classe :** Prédire la classe d'un point donné en se basant sur la classe majoritaire parmi ses `k` plus proches voisins.

### Résultats :

- La performance varie avec le choix de `k`, avec `k=1` donnant la meilleure précision de 94,75%.
- Pour `k=5` on obitient une précision de 92%.
- Pour `k=10` on obtient une précision de 88%.








## Régression Logistique Multivariée

Implémentation d'un modèle de Régression Logistique Multivariée (RLM) pour la classification des données en utilisant PyTorch.

### Étapes :

- **Préparation des Données :** Chargement des données en utilisant pandas, division en ensembles d'entraînement et de validation, et normalisation.
- **Conversion en Tenseur :** Conversion des dataframes en tenseurs PyTorch pour l'entrée du modèle.
- **Définition du Modèle :** Définition d'une classe `Reg` héritant de `nn.Module` avec une couche linéaire pour effectuer la transformation linéaire des données d'entrée.

### Résultats :

- Réglage du taux d'apprentissage (`eta`) à 0,01 et variation du nombre d'itérations montre une amélioration de la précision jusqu'à 92% pour 1000 itérations.






## Réseau de Neurones

L'approche suit des étapes similaires à la Régression Logistique Multivariée mais avec une structure de modèle différente.

### Définition du Modèle :

- La classe `NeuralNetwork` hérite de `nn.Module`.
- **Constructeur :** Accepte le nombre de caractéristiques d'entrée et le nombre de classes à prédire. Il initialise une couche linéaire, une fonction d'activation ReLU, et une dernière couche linéaire pour les logits de classe.
- **Méthode Forward :** Définit le passage en avant à travers les couches du réseau.

### Résultats :

- La précision s'améliore avec le nombre d'itérations, atteignant 93% pour 1000 itérations.

