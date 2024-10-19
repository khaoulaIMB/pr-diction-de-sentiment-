 from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import col
from pymongo import MongoClient
# Initialisation de Spark Session avec MongoDB connector
spark = SparkSession.builder.appName("Real-Time Twitter Sentiment Prediction").getOrCreate()

# Chargement du modèle entraîné
model_path = "/opt/bitnami/spark/models/twitter_sentiment_model"
model = PipelineModel.load(model_path)

# Chargement des données de validation
validation_data_path = "/opt/bitnami/spark/data/twitter_validation.csv"  # Mettez à jour le chemin vers votre fichier de validation
df_validation = spark.read.csv(validation_data_path, header=False, inferSchema=True)
df_validation = df_validation.toDF("Tweet ID", "Entity", "Sentiment", "Tweet content")

# Nettoyage des données : suppression des lignes avec des valeurs manquantes
df_validation = df_validation.na.drop()

# Prédiction des sentiments des tweets
predictions = model.transform(df_validation)

# Sélection des colonnes pertinentes pour l'affichage
results = predictions.select("Tweet ID", "Entity", "Tweet content", "prediction")
results.show()

# Conversion du DataFrame Spark en liste de dictionnaires
results_list = [row.asDict() for row in results.collect()]

# Vérification du contenu des résultats
print("Résultats à insérer dans MongoDB :")
for result in results_list:
    print(result)

# Connexion à MongoDB
client = MongoClient("mongodb://host.docker.internal:27017")
db = client["bigData"]
collection = db["predictions"]

# Insertion des résultats dans MongoDB
if results_list:
    try:
        collection.insert_many(results_list)
        print("Résultats insérés avec succès dans MongoDB.")
    except Exception as e:
        print(f"Erreur lors de l'insertion des résultats dans MongoDB : {e}")
else:
    print("Aucun résultat à insérer dans MongoDB.")

# Fermeture de la session Spark
spark.stop()

# Fermeture de la connexion MongoDB
client.close()
