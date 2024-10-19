from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
import nltk
from pyspark.ml import PipelineModel
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import os 
# Initialisation de Spark Session
spark = SparkSession.builder.appName("Twitter Sentiment Analysis").getOrCreate()

# Chargement des données
data_path = "/opt/bitnami/spark/data/twitter_training.csv"
df = spark.read.csv(data_path, header=False, inferSchema=True)
df = df.toDF("Tweet ID", "Entity", "Sentiment", "Tweet content")

# Nettoyage et préparation des données
df = df.na.drop()
stringIndexer = StringIndexer(inputCol="Sentiment", outputCol="label")
tokenizer = Tokenizer(inputCol="Tweet content", outputCol="words")
stop_words = set(stopwords.words('english'))
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", stopWords=list(stop_words))
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures")
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Régression logistique pour classification multiclasse
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.01, family="multinomial")

# Construction du pipeline
pipeline = Pipeline(stages=[stringIndexer, tokenizer, remover, hashingTF, idf, lr])

# Entraînement du modèle
model = pipeline.fit(df)

# Évaluation du modèle
predictions = model.transform(df)
evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# Sauvegarde du modèle
# Sauvegarde du modèle avec overwrite
model.write().overwrite().save("/opt/bitnami/spark/models/twitter_sentiment_model")

# Arrêt de la session Spark
spark.stop()

