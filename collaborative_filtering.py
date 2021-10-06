# -*- coding: utf-8 -*-
"""Collaborative Filtering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MbwAmboDXvmLtp9tW9_OiTsJPS8jS__u

# Collaborative Filtering

### Setup

Let's setup Spark on Colab environment.  Run the cell below!
"""

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

"""Now we authenticate a Google Drive client to download the filea we will be processing in the Spark job."""

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

id='1QtPy_HuIMSzhtYllT3-WeM3Sqg55wK_D'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.training')

id='1ePqnsQTJRRvQcBoF2EhoPU8CU1i5byHK'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.test')

id='1ncUBWdI5AIt3FDUJokbMqpHD2knd5ebp'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.item')

"""If you executed the cells above, you should be able to see the dataset I will use for this Colab under the "Files" tab on the left panel.

Next, I will import some of the common libraries needed for the task.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf

"""Let's initialize the Spark context."""

# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

"""We can easily check the current version and get the link of the web interface. In the Spark UI, we can monitor the progress of the job and debug the performance bottlenecks (if Colab is running with a **local runtime**)."""

spark

"""If we are running this Colab on the Google hosted runtime, the cell below will create a *ngrok* tunnel which will allow us to still check the Spark UI."""

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

"""### Data Loading

In this Colab, I will be using the [MovieLens dataset](https://grouplens.org/datasets/movielens/), specifically the 100K dataset (which contains in total 100,000 ratings from 1000 users on ~1700 movies).

We load the ratings data in a 80%-20% ```training```/```test``` split, while the ```items``` dataframe contains the movie titles associated to the item identifiers.
"""

schema_ratings = StructType([
    StructField("user_id", IntegerType(), False),
    StructField("item_id", IntegerType(), False),
    StructField("rating", IntegerType(), False),
    StructField("timestamp", IntegerType(), False)])

schema_items = StructType([
    StructField("item_id", IntegerType(), False),
    StructField("movie", StringType(), False)])

training = spark.read.option("sep", "\t").csv("MovieLens.training", header=False, schema=schema_ratings)
test = spark.read.option("sep", "\t").csv("MovieLens.test", header=False, schema=schema_ratings)
items = spark.read.option("sep", "|").csv("MovieLens.item", header=False, schema=schema_items)

training.printSchema()

items.printSchema()

"""### Building ALS model

Let's compute some stats!  What is the number of ratings in the training and test dataset? How many movies are in our dataset?
"""

print( "Number of ratings: ", training.count() )
print( "Number of movies: ", items.count() )

"""Using the training set, I will train a model with the Alternating Least Squares method available in the Spark MLlib: [https://spark.apache.org/docs/latest/ml-collaborative-filtering.html](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="item_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

"""Now lets compute the RMSE on the test dataset.

"""

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

"""At this point, we can use the trained model to produce the top-K recommendations for each user."""

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

ratings = training.union(test)

# Generate top 10 movie recommendations for a specified set of users
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)

userRecs.printSchema()

movieRecs.printSchema()

userRecs.show()

flat_userRecs = userRecs.withColumn("recommendations", explode(col("recommendations"))).select('user_id', 'recommendations.*')
flat_userRecs.show()

# lets read the recommendations more meaningful
flat_userRecs.join(items, flat_userRecs.item_id == items.item_id).show()

"""We can tune the ALS model by changing the maxIter, regParam which will give us a better recommendations but it will take more execution time to train the model."""

sc.stop()