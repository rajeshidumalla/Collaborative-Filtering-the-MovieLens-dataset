# Collaborative Filtering

### Setup

Let's setup Spark on Colab environment.  Run the cell below!


```python
!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

    Collecting pyspark
      Downloading pyspark-3.1.2.tar.gz (212.4 MB)
    [K     |████████████████████████████████| 212.4 MB 66 kB/s 
    [?25hCollecting py4j==0.10.9
      Downloading py4j-0.10.9-py2.py3-none-any.whl (198 kB)
    [K     |████████████████████████████████| 198 kB 61.3 MB/s 
    [?25hBuilding wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25l[?25hdone
      Created wheel for pyspark: filename=pyspark-3.1.2-py2.py3-none-any.whl size=212880768 sha256=9fcf289ae0341c75b4729c6d47f04d14f9bcec32068d7a436757f5e90de28066
      Stored in directory: /root/.cache/pip/wheels/a5/0a/c1/9561f6fecb759579a7d863dcd846daaa95f598744e71b02c77
    Successfully built pyspark
    Installing collected packages: py4j, pyspark
    Successfully installed py4j-0.10.9 pyspark-3.1.2
    The following additional packages will be installed:
      openjdk-8-jre-headless
    Suggested packages:
      openjdk-8-demo openjdk-8-source libnss-mdns fonts-dejavu-extra
      fonts-ipafont-gothic fonts-ipafont-mincho fonts-wqy-microhei
      fonts-wqy-zenhei fonts-indic
    The following NEW packages will be installed:
      openjdk-8-jdk-headless openjdk-8-jre-headless
    0 upgraded, 2 newly installed, 0 to remove and 37 not upgraded.
    Need to get 36.5 MB of archives.
    After this operation, 143 MB of additional disk space will be used.
    Selecting previously unselected package openjdk-8-jre-headless:amd64.
    (Reading database ... 155047 files and directories currently installed.)
    Preparing to unpack .../openjdk-8-jre-headless_8u292-b10-0ubuntu1~18.04_amd64.deb ...
    Unpacking openjdk-8-jre-headless:amd64 (8u292-b10-0ubuntu1~18.04) ...
    Selecting previously unselected package openjdk-8-jdk-headless:amd64.
    Preparing to unpack .../openjdk-8-jdk-headless_8u292-b10-0ubuntu1~18.04_amd64.deb ...
    Unpacking openjdk-8-jdk-headless:amd64 (8u292-b10-0ubuntu1~18.04) ...
    Setting up openjdk-8-jre-headless:amd64 (8u292-b10-0ubuntu1~18.04) ...
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/orbd to provide /usr/bin/orbd (orbd) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/servertool to provide /usr/bin/servertool (servertool) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/tnameserv to provide /usr/bin/tnameserv (tnameserv) in auto mode
    Setting up openjdk-8-jdk-headless:amd64 (8u292-b10-0ubuntu1~18.04) ...
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/idlj to provide /usr/bin/idlj (idlj) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/wsimport to provide /usr/bin/wsimport (wsimport) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/jsadebugd to provide /usr/bin/jsadebugd (jsadebugd) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/native2ascii to provide /usr/bin/native2ascii (native2ascii) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/javah to provide /usr/bin/javah (javah) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/hsdb to provide /usr/bin/hsdb (hsdb) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/clhsdb to provide /usr/bin/clhsdb (clhsdb) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/extcheck to provide /usr/bin/extcheck (extcheck) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/schemagen to provide /usr/bin/schemagen (schemagen) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/xjc to provide /usr/bin/xjc (xjc) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/jhat to provide /usr/bin/jhat (jhat) in auto mode
    update-alternatives: using /usr/lib/jvm/java-8-openjdk-amd64/bin/wsgen to provide /usr/bin/wsgen (wsgen) in auto mode


Now we authenticate a Google Drive client to download the filea we will be processing in the Spark job.


```python
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
```


```python
id='1QtPy_HuIMSzhtYllT3-WeM3Sqg55wK_D'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.training')

id='1ePqnsQTJRRvQcBoF2EhoPU8CU1i5byHK'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.test')

id='1ncUBWdI5AIt3FDUJokbMqpHD2knd5ebp'
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('MovieLens.item')
```

If you executed the cells above, you should be able to see the dataset I will use for this Colab under the "Files" tab on the left panel.

Next, I will import some of the common libraries needed for the task.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
```

Let's initialize the Spark context.


```python
# create the session
conf = SparkConf().set("spark.ui.port", "4050")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
```

We can easily check the current version and get the link of the web interface. In the Spark UI, we can monitor the progress of the job and debug the performance bottlenecks (if Colab is running with a **local runtime**).


```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://805b744e894f:4050">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.1.2</code></dd>
      <dt>Master</dt>
        <dd><code>local[*]</code></dd>
      <dt>AppName</dt>
        <dd><code>pyspark-shell</code></dd>
    </dl>
</div>

    </div>




If we are running this Colab on the Google hosted runtime, the cell below will create a *ngrok* tunnel which will allow us to still check the Spark UI.


```python
!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 4050 &')
!curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

    --2021-10-06 01:43:23--  https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    Resolving bin.equinox.io (bin.equinox.io)... 52.202.168.65, 54.161.241.46, 18.205.222.128, ...
    Connecting to bin.equinox.io (bin.equinox.io)|52.202.168.65|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 13832437 (13M) [application/octet-stream]
    Saving to: ‘ngrok-stable-linux-amd64.zip’
    
    ngrok-stable-linux- 100%[===================>]  13.19M  5.86MB/s    in 2.3s    
    
    2021-10-06 01:43:26 (5.86 MB/s) - ‘ngrok-stable-linux-amd64.zip’ saved [13832437/13832437]
    
    Archive:  ngrok-stable-linux-amd64.zip
      inflating: ngrok                   
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
    IndexError: list index out of range


### Data Loading

In this Colab, I will be using the [MovieLens dataset](https://grouplens.org/datasets/movielens/), specifically the 100K dataset (which contains in total 100,000 ratings from 1000 users on ~1700 movies).

We load the ratings data in a 80%-20% ```training```/```test``` split, while the ```items``` dataframe contains the movie titles associated to the item identifiers.


```python
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
```


```python
training.printSchema()
```

    root
     |-- user_id: integer (nullable = true)
     |-- item_id: integer (nullable = true)
     |-- rating: integer (nullable = true)
     |-- timestamp: integer (nullable = true)
    



```python
items.printSchema()
```

    root
     |-- item_id: integer (nullable = true)
     |-- movie: string (nullable = true)
    


### Building ALS model

Let's compute some stats!  What is the number of ratings in the training and test dataset? How many movies are in our dataset?


```python
print( "Number of ratings: ", training.count() )
print( "Number of movies: ", items.count() )
```

    Number of ratings:  80000
    Number of movies:  1682


Using the training set, I will train a model with the Alternating Least Squares method available in the Spark MLlib: [https://spark.apache.org/docs/latest/ml-collaborative-filtering.html](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)


```python
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="item_id", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)
```

Now lets compute the RMSE on the test dataset.



```python
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
```

    Root-mean-square error = 1.1264867353533161


At this point, we can use the trained model to produce the top-K recommendations for each user.


```python
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
```


```python
userRecs.printSchema()
```

    root
     |-- user_id: integer (nullable = false)
     |-- recommendations: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- item_id: integer (nullable = true)
     |    |    |-- rating: float (nullable = true)
    



```python
movieRecs.printSchema()
```

    root
     |-- item_id: integer (nullable = false)
     |-- recommendations: array (nullable = true)
     |    |-- element: struct (containsNull = true)
     |    |    |-- user_id: integer (nullable = true)
     |    |    |-- rating: float (nullable = true)
    



```python
userRecs.show()
```

    +-------+--------------------+
    |user_id|     recommendations|
    +-------+--------------------+
    |    471|[{1036, 12.422939...|
    |    463|[{793, 5.794737},...|
    |    833|[{1368, 6.2343984...|
    |    496|[{1240, 7.7203727...|
    |    148|[{1205, 10.957403...|
    |    540|[{1643, 6.031493}...|
    |    392|[{1615, 6.611747}...|
    |    243|[{1615, 6.749518}...|
    |    623|[{1473, 8.838204}...|
    |    737|[{1120, 9.00234},...|
    |    897|[{1643, 6.112293}...|
    |    858|[{793, 8.010915},...|
    |     31|[{853, 10.205002}...|
    |    516|[{624, 8.397214},...|
    |    580|[{1069, 9.959866}...|
    |    251|[{1278, 6.5018854...|
    |    451|[{1001, 7.89084},...|
    |     85|[{1473, 5.3242583...|
    |    137|[{962, 9.554027},...|
    |    808|[{1021, 8.569424}...|
    +-------+--------------------+
    only showing top 20 rows
    



```python
flat_userRecs = userRecs.withColumn("recommendations", explode(col("recommendations"))).select('user_id', 'recommendations.*')
flat_userRecs.show()
```

    +-------+-------+---------+
    |user_id|item_id|   rating|
    +-------+-------+---------+
    |    471|   1036|12.422939|
    |    471|   1664|  9.98227|
    |    471|    958| 9.943821|
    |    471|    440| 9.678638|
    |    471|    793| 9.591538|
    |    471|    889| 9.442203|
    |    471|    394| 8.993095|
    |    471|   1066| 8.941356|
    |    471|   1062| 8.351093|
    |    471|   1185| 8.307676|
    |    463|    793| 5.794737|
    |    463|   1466|5.6591215|
    |    463|   1113| 5.562354|
    |    463|   1282|  5.51689|
    |    463|    262| 5.374179|
    |    463|   1192| 5.339427|
    |    463|    889|5.2256794|
    |    463|    909| 5.094523|
    |    463|    980|4.9581914|
    |    463|    613|4.8710012|
    +-------+-------+---------+
    only showing top 20 rows
    



```python
# lets read the recommendations more meaningful
flat_userRecs.join(items, flat_userRecs.item_id == items.item_id).show()
```

    +-------+-------+---------+-------+--------------------+
    |user_id|item_id|   rating|item_id|               movie|
    +-------+-------+---------+-------+--------------------+
    |    471|   1036|12.422939|   1036|Drop Dead Fred (1...|
    |    471|   1664|  9.98227|   1664|8 Heads in a Duff...|
    |    471|    958| 9.943821|    958|To Live (Huozhe) ...|
    |    471|    440| 9.678638|    440|Amityville II: Th...|
    |    471|    793| 9.591538|    793|     Crooklyn (1994)|
    |    471|    889| 9.442203|    889|Tango Lesson, The...|
    |    471|    394| 8.993095|    394|Radioland Murders...|
    |    471|   1066| 8.941356|   1066|        Balto (1995)|
    |    471|   1062| 8.351093|   1062|Four Days in Sept...|
    |    471|   1185| 8.307676|   1185|In the Army Now (...|
    |    463|    793| 5.794737|    793|     Crooklyn (1994)|
    |    463|   1466|5.6591215|   1466|Margaret's Museum...|
    |    463|   1113| 5.562354|   1113|Mrs. Parker and t...|
    |    463|   1282|  5.51689|   1282|Grass Harp, The (...|
    |    463|    262| 5.374179|    262|In the Company of...|
    |    463|   1192| 5.339427|   1192|Boys of St. Vince...|
    |    463|    889|5.2256794|    889|Tango Lesson, The...|
    |    463|    909| 5.094523|    909|Dangerous Beauty ...|
    |    463|    980|4.9581914|    980| Mother Night (1996)|
    |    463|    613|4.8710012|    613|My Man Godfrey (1...|
    +-------+-------+---------+-------+--------------------+
    only showing top 20 rows
    


We can tune the ALS model by changing the maxIter, regParam which will give us a better recommendations but it will take more execution time to train the model.


```python
sc.stop()
```
