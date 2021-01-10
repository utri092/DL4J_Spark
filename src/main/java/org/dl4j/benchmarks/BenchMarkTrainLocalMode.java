package org.dl4j.benchmarks;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.launcher.SparkLauncher;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.util.List;

public class BenchMarkTrainLocalMode {

    public static void main(String[] args) throws Exception {
        int warmup = 5;
        int iterations = 100;
        int numFeaturesFromFirstColumn = 2;   // No of feature columns from the start to end of row
        int numLabelsAfterLastFeatureColumn = 1;    // No of labels after the last feature column
        int batchsize = 2048; // Not used yet
        int numEpochs = 50;

        SparkConf conf = new SparkConf();
        conf.setAppName("BenchMarkTrainLocalMode");
        conf.setMaster("local[*]");



        JavaSparkContext sc = new JavaSparkContext(conf);

       /* SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DL4J Spark Imagenet Classifier");
        conf.set("spark.locality.wait","0");
        conf.set("spark.executor.extraJavaOptions","-Dorg.bytedeco.javacpp.maxbytes=6G -Dorg.bytedeco.javacpp.maxphysicalbytes=6G");
        conf.set(SparkLauncher.DRIVER_EXTRA_JAVA_OPTIONS,"-Dorg.bytedeco.javacpp.maxbytes=6G -Dorg.bytedeco.javacpp.maxphysicalbytes=6G");*/

//        JavaSparkContext sc = new JavaSparkContext(conf);

        /*VoidConfiguration voidConfiguration = VoidConfiguration.builder()
//                .controllerAddress(masterIP)
                .controllerAddress("192.168.137.226")
                .unicastPort(40123)                          // Port number that should be open for IN/OUT communications on all Spark nodes
                *//*  .networkMask("192.168.0.0/16")                   // Local network mask
                .controllerAddress("192.168.0.139")                // IP address of the master/driver node
                .meshBuildMode(MeshBuildMode.PLAIN)*//*
                .build();*/

        /*TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchsize)
                .rngSeed(12345)
                .collectTrainingStats(false)
                .batchSizePerWorker(batchsize)              // Minibatch size for each worker
                .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(1E-3))     //Threshold algorithm determines the encoding threshold to be use.
                .workersPerNode(1)          // Workers per node
                .build();*/

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/benchmarks/model.h5");

        String filePath = "./src/main/resources/benchmarks/dataset-1_converted.csv";
        JavaRDD<String> rddString = sc.textFile(filePath);
        RecordReader recordReader = new CSVRecordReader(0, ',');
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));


        JavaRDD<DataSet> trainingData = rddWritables.map(new DataVecDataSetFunction(numFeaturesFromFirstColumn, numLabelsAfterLastFeatureColumn, true, null, null));

        // very basic need to explore further
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .batchSizePerWorker(batchsize)
                .averagingFrequency(5).workerPrefetchNumBatches(2)
                .build();


        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, tm);

//        System.out.println(model.getLayers());
//        System.out.println(model.getLayerWiseConfigurations());


        for(int j = 0; j < warmup; j++){
            for (int i = 0; i < numEpochs; i++) {
                // @note: For Hadoop HDFS direct pass using fitpaths() should be possible from docs
                //       sparkNet.fit("./src/main/resources/datasets/dataset-1_converted.csv");
                sparkNet.fit(trainingData);
            }
        }

        System.out.println("Before Train!");

        long startTime = System.nanoTime();

        for(int j = 0; j < iterations; j++){
            for (int i = 0; i < numEpochs; i++) {
                // @note: For Hadoop HDFS direct pass using fitpaths() should be possible from docs
                //       sparkNet.fit("./src/main/resources/datasets/dataset-1_converted.csv");
                sparkNet.fit(trainingData);
            }
        }
        long endTime = System.nanoTime();

        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds
        System.out.println(duration/1000000000);
        System.out.println("DONE TRAINING");


    }

}
