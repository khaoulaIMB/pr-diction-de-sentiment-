FROM bitnami/spark:latest

USER root

# Installation de Python, pip et des packages nécessaires
RUN install_packages python3-pip
RUN pip install --upgrade pip
RUN pip install jupyter pyspark nltk py4j numpy pymongo  # Ajoutez pymongo ici

# Télécharger les données nécessaires de NLTK
RUN python -m nltk.downloader popular

# Autoriser l'accès au port Jupyter
EXPOSE 8888
ENV SPARK_SUBMIT_ARGS="--packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 pyspark-shell"

# Commande pour démarrer Jupyter au lancement du conteneur
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

USER 1001
