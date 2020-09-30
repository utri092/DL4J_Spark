package org.deeplearning4j.examples.sample;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.io.ClassPathResource;

public class Inference {

    public static void main(String[] args) throws Exception {
//        String fullModel = new ClassPathResource(System.getProperty("user.dir") + "/model.h5").getFile().getPath();
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("model.h5");

    }

}
