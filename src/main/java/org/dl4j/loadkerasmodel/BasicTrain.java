package org.dl4j.loadkerasmodel;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

public class BasicTrain {

    public static void main(String[] args) throws Exception {

        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JTrain");
        conf.setMaster("local[*]");


        JavaSparkContext sc = new JavaSparkContext(conf);

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5", true);

        String filePath = "./src/main/resources/datasets/dataset-1_small.csv";
        JavaRDD<String> rddString = sc.textFile(filePath);
        RecordReader recordReader = new CSVRecordReader(0, ',');
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));

        int firstColumnLabel = 2;   // No of features from the start to end of row
        int lastColumnLabel = 1;    // Last column
        JavaRDD<DataSet> trainingData = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));
        // very basic need to explore further
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, tm);

        int batchsize = 32; // Not used yet
        int numEpochs = 10;


        // Collect RDD for printing. Cant print other rdds

        /*for(String line: rddString.collect()){
            System.out.println("* " + line);
        }*/

        System.out.println("Before Train!");

        for (int i = 0; i < numEpochs; i++) {
            /*@note: For Hadoop HDFS direct pass should be possible from docs*/
            //       sparkNet.fit("./src/main/resources/datasets/dataset-1_converted.csv");
            sparkNet.fit(trainingData);
        }

        System.out.println("DONE");


    }


}
