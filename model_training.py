"""
User: Mohammad-Shamel Agha
ID: ma2298
Course Number: CS 643
Date: 12-2-2023
File Name: model_training.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import findspark
import pyspark
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col
from pyspark.mllib.linalg import Vectors
from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml import Pipeline
import warnings
warnings.filterwarnings("ignore")

findspark.init()
findspark.find()

conf = pyspark.SparkConf().setAppName('winequality').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

df = spark.read.format("csv").load("TrainingDataset.csv", header=True, sep=";")
df.printSchema()
df.show(5)

for col_name in df.columns[1:-1]+['""""quality"""""']:
    df = df.withColumn(col_name, col(col_name).cast('float'))
df = df.withColumnRenamed('""""quality"""""', "quality")

features = np.array(df.select(df.columns[1:-1]).collect())
label = np.array(df.select('quality').collect())

# Function to create label points
def to_labeled_point(features, labels):
    lp = []
    for x, y in zip(features, labels):
        labeled_points = LabeledPoint(y, x)
        lp.append(labeled_points)
    return lp

# Function to convert to RDD
def to_rdd(sc, labeled_points):
    return sc.parallelize(labeled_points)

data_label_point = to_labeled_point(features, label)

data_label_point_rdd = to_rdd(sc, data_label_point)

model_train, model_test = data_label_point_rdd.randomSplit([0.7, 0.3], seed=21)

RF = RandomForest.trainClassifier(model_train, numClasses=10, categoricalFeaturesInfo={},
                                       numTrees=21, featureSubsetStrategy="auto",
                                       impurity='gini', maxDepth=30, maxBins=32)

prediction = RF.predict(model_test.map(lambda x: x.features))
prediction_rdd = model_test.map(lambda y: y.label).zip(prediction)
prediction_df = prediction_rdd.toDF()

quality_prediction = prediction_rdd.toDF(["quality", "prediction"])
quality_prediction.show()
quality_prediction_df = quality_prediction.toPandas()

print("---------------Output-----------------")
print("Accuracy : ", accuracy_score(quality_prediction_df['quality'], quality_prediction_df['prediction']))
print("F1- score : ", f1_score(quality_prediction_df['quality'], quality_prediction_df['prediction'], average='weighted'))

test_error = predic_rdd.filter(
    lambda y: y[0] != y[1]).count() / float(model_test.count())
print('Test Error : ' + str(test_error))

RF.save(sc, 's3://pa2bucket/trainingmodel.model')