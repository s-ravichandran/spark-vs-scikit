''' Pyspark Template file '''
from __future__ import print_function
import sys
from pyspark.sql import SparkSession
import xml.etree.ElementTree as ET
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.ml.classification import LogisticRegression
import re
import time
import math

# User-defined function to extract Id, Body, Tags
def extract (s):
    if(not s.startswith('<?xml') and not s.startswith('<posts>') and not s.endswith('</posts>')):
        data = ET.fromstring(s)
        Id = data.attrib['Id']
        Body = data.attrib['Body']
        if('Tags' in data.attrib):
            Tags = data.attrib['Tags']
        else:
            Tags = None
            Taglist = Tags
        Body = re.sub('<\\S+>','', Body)
        Body = re.sub('\n',' ', Body)
        if (Tags):
            Tags = re.sub('<','',Tags)
            Tags = re.sub('>',',',Tags)
            Taglist = Tags.split(',')[:-1]
        return Row(Id=Id, Body=Body, Tags=Taglist)
    else:
        #print (s)
        return Row(Id=None, Body=None, Tags=None)

def label (row, targetTag):
    if (row['Tags']):
        if (targetTag in row['Tags']):
            label = 1.0
        else:
            label = 0.0
    else:
        label = None
    return Row(Id=row['Id'], Body=row['Body'], Tags=row['Tags'], Label=label)

def roundup(x):
    return Row(Body=x['Body'], Id=x['Id'], Label=x['Label'], Tags=x['Tags'], tokens=x['tokens'], Features=x['Features'], rawPrediction=x['rawPrediction'], probability=x['probability'], Prediction=round(x['Prediction']))

# Setup SparkConf and SparkContext
try:
    ss = SparkSession.builder.getOrCreate()
except:
    print ('Error creating SparkSession')
    sys.exit(1)

''' Do what you need here '''

# Load XML File
datDF = ss.read.text('hdfs://namenode:9000/user/root/Posts.xml')

# Open a log file
logfile_name = 'log_' + sys.argv[1] +'.txt'
logfile = open(logfile_name, 'w+')


num_lines = int(sys.argv[1])
# Create a new DataFrame with Id, Body and Tags
dataDF = datDF.limit(num_lines)
start_extract = time.time()
processedDF = dataDF.rdd.map(lambda x: extract(x[0].encode('utf-8'))).toDF()
#processedDF = dataDF.foreach(extract)
finish_extract = time.time()
extraction_time = finish_extract - start_extract
extraction_log = 'Time spent to apply extract() = ' + str(extraction_time) + ' seconds'
logfile.write(extraction_log)
print (extraction_log)

# Add a Label column to create 'training data'
targetTag = 'java'
start_label = time.time()
modDF = processedDF.rdd.map(lambda x: label(x, targetTag)).toDF()
#modDF = processedDF.foreach(label)
finish_label = time.time()
label_time = finish_label - start_label
label_log = 'Time to apply label() ' + str(label_time) + ' seconds'
logfile.write(label_log)
print (label_log)
#
start_removeNull = time.time()
labeledDF = modDF.where(modDF.Label.isNotNull())
finish_removeNull = time.time()
removeNull_time = finish_removeNull - start_removeNull
null_log = 'Time to remove null labels ' + str(removeNull_time) + ' seconds'
logfile.write(null_log)
print (null_log)
#

tok = Tokenizer(inputCol='Body', outputCol='tokens')
tf = HashingTF(numFeatures=60000, inputCol = tok.getOutputCol(), outputCol='Features')
lr = LogisticRegression(regParam=0.02, featuresCol='Features', labelCol='Label', maxIter=5, predictionCol='Prediction')
pipeline = Pipeline(stages=[tok, tf, lr])
start_split = time.time()
trainDF, testDF = labeledDF.randomSplit([0.8, 0.2])
finish_split = time.time()
split_time = finish_split - start_split
split_log = 'Time to split data ' + str(split_time) +' seconds'
logfile.write(split_log)
print (split_log)

#
start_fit = time.time()
model = pipeline.fit(trainDF)
finish_fit = time.time()
fit_time = finish_fit - start_fit
fit_log = 'Time to fit ' + str(fit_time) + ' seconds'
logfile.write(fit_log)
print (fit_log)

logfile.close()
#model.save('model')

'''
#
start_predictions = time.time()
predictionsDF = model.transform(testDF)
finish_predictions = time.time()
predictions_time = finish_predictions - start_predictions
print ('Time to predict ' + str(predictions_time) + ' seconds')


start_agg = time.time()
predictions_count = predictionsDF.count()
roundedDF = predictionsDF.rdd.map(lambda x: roundup(x)).toDF()
correct_count = roundedDF.where(roundedDF.Prediction == 1.0).join(testDF, roundedDF.Prediction == testDF.Label).count()
finish_agg = time.time()
aggregate_time = finish_agg - start_agg


print (correct_count)
print (predictions_count)
print ('Accuracy: ' + str(float(correct_count)/predictions_count))
print()
print ('Time Summary')
print ('Time spent to apply extract() = ' + str(extraction_time) + ' seconds')
print ('Time to apply label() ' + str(label_time) + ' seconds')
print ('Time to remove null labels ' + str(removeNull_time) + ' seconds')
print ('Time to split data ' + str(split_time) +' seconds')
print ('Time to fit ' + str(fit_time) + ' seconds')
print ('Time to predict ' + str(predictions_time) + ' seconds')
print ('Time to calculate accuracy (aggregation) ' + str(aggregate_time) + ' seconds')
'''
''' Stop the SparkContext'''
ss.stop()
print ("Spark session closed.")
