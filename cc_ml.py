import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# create spark session
spark = SparkSession.builder.appName('cc_ml').getOrCreate()

# create DataFrame by using spark session to read csv, cast "string floats" to floats, drop nulls
df = (spark.read
          .format("csv")
          .option('header', 'true')
          .load("/Users/school/Downloads/CC.csv"))

dataset = df.select(col('Class').cast('float'),
                         col('Time').cast('float'),
                         col('V1').cast('float'),
                         col('V2').cast('float'),
                         col('V3').cast('float'),
                         col('V4').cast('float'),
                         col('V5').cast('float'),
                         col('V6').cast('float'),
                         col('V7').cast('float'),
                         col('V8').cast('float'),
                         col('V9').cast('float'),
                         col('V10').cast('float'),
                         col('V11').cast('float'),
                         col('V12').cast('float'),
                         col('V13').cast('float'),
                         col('V14').cast('float'),
                         col('V15').cast('float'),
                         col('V16').cast('float'),
                         col('V17').cast('float'),
                         col('V18').cast('float'),
                         col('V19').cast('float'),
                         col('V20').cast('float'),
                         col('V21').cast('float'),
                         col('V22').cast('float'),
                         col('V23').cast('float'),
                         col('V24').cast('float'),
                         col('V25').cast('float'),
                         col('V26').cast('float'),
                         col('V27').cast('float'),
                         col('V28').cast('float'),
                         col('Amount').cast('float'),
                        )
print(dataset.count())
# create new column 'features' that contains vector of all features for each row
required_features = ['Time',
                    'V1',
                    'V2',
                    'V3',
                    'V4',
                    'V5',
                    'V6',
                    'V7',
                    'V8',
                    'V9',
                    'V10',
                    'V11',
                    'V12',
                    'V13',
                    'V14',
                    'V15',
                    'V16',
                    'V17',
                    'V18',
                    'V19',
                    'V20',
                    'V21',
                    'V22',
                    'V23',
                    'V24',
                    'V25',
                    'V26',
                    'V27',
                    'V28',
                    'Amount'
                   ]
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(dataset)
(training_data, test_data) = transformed_data.randomSplit([0.3,0.7])

# fit Random Forest Classifier Estimator onto data, create transformative model by using Estimator on training, generate
# predictions using transformative model on test
rf = RandomForestClassifier(labelCol='Class', 
                            featuresCol='features')
model = rf.fit(training_data)
predictions = model.transform(test_data)

# evaluate accuracy of predictions
evaluator = MulticlassClassificationEvaluator(
    labelCol='Class', 
    predictionCol='prediction', 
    metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
tp = evaluator.evaluate(predictions, {evaluator.metricName: "truePositiveRateByLabel",
    evaluator.metricLabel: 1.0})
fp = 1 - tp
tn = evaluator.evaluate(predictions, {evaluator.metricName: "truePositiveRateByLabel",
    evaluator.metricLabel: 0.0})
fn = 1 - tn
print('Accuracy =', accuracy)
print('------------------')
print('Sensitivity =', tp/(tp+fn))
print('Specificity =', tn/(tn+fp))
print('------------------')
print('True Positive =', tp)
print('False Positive =', fp)
print('True Negative =', tn)
print('False Negative =', fn)
