# machineLearningProjects

Collection de notebooks Jupyter autour de sujets classiques de machine learning et de deep learning: classification tabulaire, NLP, CNN, RNN et recherche d'hyperparametres.

## Contenu

- `classificationDecisionTree.ipynb`, `random_forest.ipynb`, `mlp_classification.ipynb`: classification du diabete avec `diabetes.csv`.
- `cnn_tensorflow.ipynb`, `batch_normalization.ipynb`, `dropout_cnn.ipynb`: reseaux convolutifs TensorFlow/Keras sur MNIST.
- `textMining.ipynb`, `imdb_textMining.ipynb`: preprocessing texte, vectorisation Bag of Words / TF-IDF et classification.
- `word_emnedded.ipynb`, `imdn_conv1d.ipynb`, `rnn_imdb.ipynb`: embeddings, Conv1D et RNN pour l'analyse de sentiment IMDB.
- `weather_condition.ipynb`: modelisation de series temporelles meteo avec le dataset Jena Climate.
- `tensorflow_Grid_Random_Search.ipynb`: exemple de recherche d'hyperparametres avec TensorFlow, SciKeras et scikit-learn.
- `Helper/helper.py`: fonction utilitaire pour visualiser les frontieres de decision.

## Donnees

Le depot contient les datasets utilises par les notebooks:

- `diabetes.csv`: donnees medicales tabulaires avec la cible `Outcome`.
- `IMDB Dataset.csv`: critiques de films et labels de sentiment.
- `jena_climate_2009_2016.csv`: mesures meteorologiques horodatees.

## Installation

Creer un environnement Python, puis installer les dependances principales:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras scikeras nltk beautifulsoup4 prettytable jupyter
```

Certains notebooks NLP peuvent necessiter des ressources NLTK:

```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
```

## Utilisation

Lancer Jupyter depuis la racine du depot:

```bash
jupyter notebook
```

Ouvrir ensuite le notebook souhaite. Les chemins vers les CSV sont relatifs a la racine du depot, donc il est preferable de lancer Jupyter depuis ce dossier.

