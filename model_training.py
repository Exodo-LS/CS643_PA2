"""
User: Mohammad-Shamel Agha
ID: ma2298
Course Number: CS 643
Date: 12-2-2023
File Name: model_training.py
"""

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col


conf = (SparkConf().setAppName("Programming Assignment 2: Model Training"))
sc = SparkContext("local", conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://pa2bucket/TrainingDataset.csv')
vf = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true', sep=';').load('s3://pa2bucket/ValidationDataset.csv')

# dropping quality column
dFrame = df.select(df.columns[:11])

output = df.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

model = RandomForest.trainClassifier(output,numClasses=10,categoricalFeaturesInfo={}, numTrees=60, maxBins=32, maxDepth=4, seed=42)


validation = vf.rdd.map(lambda row: LabeledPoint(row[-1], Vectors.dense(row[:11])))

predictions = model.predict(validation.map(lambda x: x.features))
labels = validation.map(lambda lp: lp.label).zip(predictions)

metrics = MulticlassMetrics(labels)

# Overall statistics
f1Score = metrics.fMeasure()
print("==== Summary Statistics ====")
print("Weighted F(1) Score = %3s" % metrics.weightedFMeasure())

print("\n\n==== Saving model ====")
#Saving model
model.save(sc, "s3://pa2bucket/training_model")

print("Model Saved successfully")