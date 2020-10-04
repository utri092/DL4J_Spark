package org.dl4j.benchmarks;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class CreatePartitionsFromDataset {

    public static void main(String[] args) throws Exception {

        SparkConf conf = new SparkConf();
        conf.setAppName("DL4JTrain");
        conf.setMaster("local[4]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        String filePath = "./src/main/resources/benchmarks/dataset-1_converted.csv";
        JavaRDD<String> rddstring = sc.textFile(filePath);


    }
}
