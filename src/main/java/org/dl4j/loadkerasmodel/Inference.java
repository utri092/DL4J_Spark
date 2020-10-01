package org.dl4j.loadkerasmodel;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
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
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import java.util.List;


public class Inference {

    public static void main(String[] args) throws Exception {
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5");



        String filePath = "./src/main/resources/datasets/predictions.csv";




        //RecordReader rr = new CSVRecordReader(',');





        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JTinyImageNetSparkPreproc");
        conf.setMaster("local[2]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // very basic need to explore further
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1).build();

        SparkDl4jMultiLayer net = new SparkDl4jMultiLayer(sc,model,tm);


        int batchsize =32;


        // Generate RDD
        JavaPairRDD<String, PortableDataStream> stringData =sc.binaryFiles(filePath);


        /*System.out.println(net.getNetwork().);*/


        //net.feedForwardWithKey(stringData,batchsize);


        net.fit("./src/main/resources/datasets/dataset-1_converted.csv");

        System.out.println("DONE");



    }

}
