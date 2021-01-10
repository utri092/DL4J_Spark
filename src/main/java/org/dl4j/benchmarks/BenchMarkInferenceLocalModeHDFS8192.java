package org.dl4j.benchmarks;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;


public class BenchMarkInferenceLocalModeHDFS8192 {

    public static JavaSparkContext startSparkSession(){
        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JInferenceLocalModeHDFS");
        conf.setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static SparkDl4jMultiLayer createModelFromBin(String modelPath, JavaSparkContext sc) throws IOException, URISyntaxException {
        //@detail Takes in HDFS string path and tries to get model.bin

        MultiLayerNetwork model = null;
        MultiLayerNetwork net = null;

        FileSystem fileSystem = FileSystem.get(new URI ("hdfs://afog-master:9000"), sc.hadoopConfiguration());

        try(BufferedInputStream is = new BufferedInputStream(fileSystem.open(new Path(modelPath)))){
            net = ModelSerializer.restoreMultiLayerNetwork(is);
        }

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, net, tm);



        return sparkNet;

    }

    public static JavaRDD<DataSet> extractTestDataset(String filePath, JavaSparkContext sc){
        JavaRDD<String> rddString = sc.textFile(filePath);
        RecordReader recordReader = new CSVRecordReader(0, ',');
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));


        int firstColumnLabel = 2;   // No of feature columns from the start to end of row
        int lastColumnLabel = 1;    // No of labels after the last feature column
        JavaRDD<DataSet> testData = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));

       /* for (DataSet dataSet : testData.collect()) {
            System.out.println(dataSet.getFeatures());
        }*/

        return testData;
    }

    public static JavaPairRDD<String, INDArray> makePredictions( JavaPairRDD<String, INDArray> testPairs, SparkDl4jMultiLayer sparkNet){
        // @detail Tuple2: Scala class expects two arguments. Tuple3 and Tuple4 are alternatives
        // @arg-1: Name of label/s
        // @arg-2: INDArray of features from JavaRDD<Dataset>

        JavaPairRDD<String, INDArray> predictions = sparkNet.feedForwardWithKey(testPairs,8192);

        return predictions;
    }

    public static void main(String[] args) throws Exception {
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);

        int iterations = 50;

        JavaSparkContext sc = startSparkSession();


        String localModelPath = "hdfs://afog-master:9000/part4-projects/resources/benchmarks/model.bin";

        SparkDl4jMultiLayer sparkNet = createModelFromBin(localModelPath, sc);

        String datafilePath = "hdfs://afog-master:9000/part4-projects/resources/benchmarks/dataset-1_converted.csv";

        JavaRDD<DataSet> testData = extractTestDataset(datafilePath, sc);

        System.out.println("Before Inferencing!");

        JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("carparkOccupancy", f.getFeatures()));

        long startTime = System.nanoTime();

        JavaPairRDD<String, INDArray> predictions = null;

        for(int i = 0 ; i < iterations; i++){
            predictions = makePredictions(testPairs, sparkNet);

            predictions.count();

        }

        long endTime = System.nanoTime();

        long duration = (endTime - startTime);  //divide by 1000000 to get milliseconds
        System.out.println("No of Records: " + predictions.count());
        System.out.println("Total Time in milliS: " + duration/1000000);
        System.out.println("Average Time in nanoS: " + duration / (iterations * predictions.count()));
        System.out.println("DONE Inferencing");

    }

}
