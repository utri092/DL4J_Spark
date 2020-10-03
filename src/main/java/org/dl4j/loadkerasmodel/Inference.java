package org.dl4j.loadkerasmodel;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;


public class Inference {

    public static JavaSparkContext startSparkSession(){
        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JTrain");
        conf.setMaster("local[*]");

        return new JavaSparkContext(conf);
    }

    public static SparkDl4jMultiLayer createModel(JavaSparkContext sc){
        MultiLayerNetwork model = null;
        try {
            model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5", true);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvalidKerasConfigurationException e) {
            e.printStackTrace();
        } catch (UnsupportedKerasConfigurationException e) {
            e.printStackTrace();
        }

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

//        System.out.println(model.getLayers());
//        System.out.println(model.getLayerWiseConfigurations());

        return new SparkDl4jMultiLayer(sc, model, tm);

    }

    public static JavaRDD<DataSet> extractTestDataset(String filePath, JavaSparkContext sc){
        JavaRDD<String> rddString = sc.textFile(filePath);
        RecordReader recordReader = new CSVRecordReader(0, ',');
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));


        int firstColumnLabel = 2;   // No of feature columns from the start to end of row
        int lastColumnLabel = 1;    // No of labels after the last feature column
        JavaRDD<DataSet> testData = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));

        return testData;
    }

    public static JavaPairRDD<String, INDArray> makePredictions(JavaRDD<DataSet> testData, SparkDl4jMultiLayer sparkNet){
        // @detail Tuple2: Scala class expects two arguments. Tuple3 and Tuple4 are alternatives
        // @arg-1: Name of label/s
        // @arg-2: INDArray of features from JavaRDD<Dataset>

        JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("carparkOccupancy", f.getFeatures()));

        JavaPairRDD<String, INDArray> predictions = sparkNet.feedForwardWithKey(testPairs,5);

        predictions.collect().forEach(p->{
            System.out.println(p._1 + ": " + p._2);
        });

        return predictions;
    }

    public static void main(String[] args) throws Exception {

        JavaSparkContext sc = startSparkSession();

        SparkDl4jMultiLayer sparkNet = createModel(sc);

        String filePath = "./src/main/resources/datasets/predictions_small.csv";

        JavaRDD<DataSet> testData = extractTestDataset(filePath, sc);

        JavaPairRDD<String, INDArray> predictions = makePredictions(testData, sparkNet);

        //@Bug Will not work if dir already exists. Throws exception which is not handled by spark
//        predictions.saveAsTextFile("./src/main/resources/inferences/");
        System.out.println("DONE");

    }

}
