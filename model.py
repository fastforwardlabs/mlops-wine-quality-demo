from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
import pandas
import time

# import the new SDK
import cdsw

spark = SparkSession.builder \
      .appName("wine-quality-model") \
      .master("local[*]") \
      .config("spark.driver.memory","4g")\
      .config("spark.hadoop.fs.s3a.aws.credentials.provider","org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")\
      .config("spark.hadoop.fs.s3a.metadatastore.impl","org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore")\
      .config("spark.hadoop.fs.s3a.delegation.token.binding","")\
      .config("spark.hadoop.yarn.resourcemanager.principal","csso_abreshears")\
      .getOrCreate()
    
model = PipelineModel.load("file:///home/cdsw/models/spark")


schema = StructType([StructField("fixedAcidity", DoubleType(), True),     
  StructField("volatileAcidity", DoubleType(), True),     
  StructField("citricAcid", DoubleType(), True),     
  StructField("residualSugar", DoubleType(), True),     
  StructField("chlorides", DoubleType(), True),     
  StructField("freeSulfurDioxide", DoubleType(), True),     
  StructField("totalSulfurDioxide", DoubleType(), True),     
  StructField("density", DoubleType(), True),     
  StructField("pH", DoubleType(), True),     
  StructField("sulphates", DoubleType(), True),     
  StructField("Alcohol", DoubleType(), True)
])


# Decorate predict function with new functionality that 
# 1) sends metrics via the track_metric() method
# 2) adds a unique identifier for tracking the outputs

@cdsw.model_metrics
def predict(args):
  split=args["feature"].split(";")
  features=[list(map(float,split[:11]))]
  features_df = spark.createDataFrame(features, schema)#.collect()
  features_list = features_df.collect()
  
  # Let's track the inputs to the model
  for x in features_list:
    cdsw.track_metric("fixedAcidity", x["fixedAcidity"])
    cdsw.track_metric("volatileAcidity", x["volatileAcidity"])
    cdsw.track_metric("citricAcid", x["citricAcid"])
    cdsw.track_metric("residualSugar", x["residualSugar"])
    cdsw.track_metric("chlorides", x["chlorides"])
    cdsw.track_metric("freeSulfurDioxide", x["freeSulfurDioxide"])
    cdsw.track_metric("totalSulfurDioxide", x["totalSulfurDioxide"])
    cdsw.track_metric("density", x["density"])
    cdsw.track_metric("pH", x["pH"])
    cdsw.track_metric("sulphates", x["sulphates"])
    cdsw.track_metric("Alcohol", x["Alcohol"])
  
  resultdf=model.transform(features_df).toPandas()["prediction"][0]
  
  if resultdf == 1.0:
    to_return = {"result": "Poor"}
  else:
    to_return = {"result" : "Excellent"}
  
  # Let's track the prediction we're making
  cdsw.track_metric("prediction", to_return["result"])
  
  return to_return
  
# pre-heat the model
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good

time.sleep(1)

predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good

time.sleep(2)

predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good

time.sleep(1)

predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad

time.sleep(3)

predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad

time.sleep(1)

predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad

time.sleep(1)


predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good
predict({"feature": "7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4"}) #bad
predict({"feature": "7.3;0.65;0.0;1.2;0.065;15.0;21.0;0.9946;3.39;0.47;10.0"}) #good