package org.dl4j.loadkerasmodel;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
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
import org.deeplearning4j.spark.util.MLLibUtil;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.util.List;


public class Inference {

    public static void main(String[] args) throws Exception {

        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JInference");
        conf.setMaster("local[*]");
        
        JavaSparkContext sc = new JavaSparkContext(conf);

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5", true);


        // very basic need to explore further
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc,model,tm);

        String testPath = "./src/main/resources/predictions_small.csv";
        JavaRDD<String> rddString = sc.textFile(testPath);
        RecordReader recordReader = new CSVRecordReader(0, ',');
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));

        int firstColumnLabel = 2;   // No of features from the start to end of row
        int lastColumnLabel = 1;    // Last column

        JavaRDD<DataSet> testData = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));
        //TODO: Distributed Inference

        /*for(DataSet d : testData.collect()){
            System.out.println("* " + d.getFeatures());
        }*/

        JavaPairRDD<INDArray, INDArray> testPairs = testData.mapToPair(f-> new Tuple2<>(f.getFeatures(), f.getLabels()));

        JavaPairRDD<INDArray, INDArray> predictions = sparkNet.feedForwardWithKey(testPairs, 5);


        //TODO: Alternate Method that accepts Matrix or Vector

        // Not Working Yet
//        sparkNet.predict(MLLibUtil.toVector(d.getFeatures())));
        System.out.println("DONE");

    }

}
