# A comparative study of Apache Spark and Scikit-learn

I took up a logistic regression task and compared the training times between Apache Spark and Scikit-learn.

## The task:
The dataset that I used is the set of posts on [Stackoverflow.com](http://stackoverflow.com/), which can be found [here](https://archive.org/details/stackexchange). The details of the file are as follows
 - Name: stackoverflow.com-Posts.7z
 - Size: 52GB
 - Description: Each line is an XML <row> tag that contains fields such as Id, Body, Title, Tags etc.
Given this dataset, the task is to predict the tags when the body of the post is given. Ideally, this should be a multi-class classification task but that is not the main focus here. So, I decided to use logistic regression to predict if a post can be tagged with the given input tag or not (a binary classification task).
 
## The setup:
For the Spark setup, I set up a cluster of 6 nodes (4 slaves, 1 namenode, 1 resource manager). I installed hadoop and started the HDFS and Yarn daemons and deployed the task with Spark on Yarn in cluster mode. I ran the application with 4 executors, with 8GB of memory and 1 CPU core per executor.
 
For the scikit-learn setup, I used a single node with 64GB of memory.

The file `spark_train.py` contains the code for loading the XML file, parsing it to extract `Id, Body, Tags` from each row. The `targetTag` is then searched for in `tags` and the row is labeled `1.0` if it is found and `0.0` otherwise. To run the file on Spark with my settings, run
```spark-submit --master yarn --deploy-mode cluster --executor-memory 8g --num-executors 1 train.py <n_rows>``` 
where `n_rows` is the number of XML rows to be used as training data.

The file `sk_train.py` contains the corresponding scikit-learn implementation. It requires `scikit-learn` and `pandas` to be installed.

## Analysis

The following are the measurements that I made for this comparison.
 - Time to extract `Id, Body, Tags` from the XML file
 - Time to fit the training data to generate a model
 I measured these for `100K, 250K, 500K` and `1M` rows of the XML file. The following are the running times (approximated to the nearest integer).
 
### Extraction Time (in seconds)
 
  | No. of rows | Spark | Scikit-learn| 
  | ------------|-------|-------------|
  | 100K        | 139   | 56          |
  | 250K        | 136   | 212         |
  | 500K        | 140   | 926         |
  | 1M          | 145   | 4598        |  
 
![Plot](https://github.com/s-ravichandran/spark_vs_scikit/blob/master/Extract.png)
 
### Fitting time (in seconds)
 
 | No. of rows | Spark | Scikit-learn|
 |-------------|-------|------------|
 | 100K        | 18   | 6          |
 | 250K        | 43   | 16         |
 | 500K        | 80   | 40         |
 | 1M          | 154   | 111       |
 
![Plot](https://github.com/s-ravichandran/spark_vs_scikit/blob/master/Fit_times.png)
