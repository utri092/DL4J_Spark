# DeepLearning4J with Spark

 The following programs were written to explore Machine Learning using DL4J and Java API of Spark. 

 The model training and inference scripts are benchmarks, with results being used for the purposes of a Research Project
 
 See each folder and script for details.
 
 ### Compile using Maven for Non-IDE Approach
 `mvn clean compile install`
 
  ### Local Mode Command Example
  
 `java -cp target/<jar-file> org.dl4j.benchmarks.BenchMarkInferenceLocalModeHDFS2048`
 
 ### Distributed Command Example
 
 `spark-submit --class org.dl4j.benchmarks.BenchMarkInferenceDistributedHDFS8192 --master spark://afog-master:7077 --conf spark.executor.memory=2g --total-executor-cores=12 --executor-cores=4 target/deeplearning4j-example-sample-1.0.0-beta7-bin.jar`
 
 ### Notes
 1) As of the DL4J version in the pom.xml, CSV format datasets need headers to be removed if using CSVRecordReader. Skipping lines does not work and is a bug.
 2) Configure ram and cores according to requirements
 3) <jar-file-name>-bin.jar contains all the dependencies. Non bin/Non Uber jar files lack them and can be used to run programs in Spark Local Mode.
 4) Read [Lazy Evaluation Article 1](https://www.alibabacloud.com/forum/read-535) and [Lazy Evaluation Article 2](https://data-flair.training/blogs/spark-rdd-operations-transformations-actions/) for understanding inserting actions in time measurement.
5) Distributed Training Code files do not work on ARM in this release. Follow issue [here](https://github.com/eclipse/deeplearning4j/issues/9110#issuecomment-756430695) for updates 
 

