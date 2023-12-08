package test.textToImage;

import data.ImageCreator;
import data.flickr.FlickrLoader3D;
import data.loaders.TransformData;
import data.network_train.NNData3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class FlickrTest {
    public static void main(String[] args) throws Exception {
        FlickrLoader3D loader = new FlickrLoader3D(8, 20000, 1000).setTransformData(new TransformData.Tanh());

        NeuralNetwork network = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/flickr/flick_network_8.txt")))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer(0.0002))
                .create();

//        NeuralNetwork network = new NeuralNetwork()
//                .addInputLayer(100)
//                .addLayer(new EmbeddingLayer(20000, 64))
//                .addLayer(new LSTMLayer(256, 0.2, false))
//                .addLayer(new FlattenLayer2D())
//                .addDenseLayer(4*4*128, new FunctionActivation.ReLU())
//                .addLayer(new ReshapeLayer3D(4, 4, 128))
//                .addLayer(new ConvolutionTransposeLayer(128, 4, 2, 1))
//                .addBatchNormalizationLayer(0.9)
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(3, 3, 1, 1))
//                .addActivationLayer(new FunctionActivation.Tanh())
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .setOptimizer(new AdamOptimizer(0.0002))
//                .create();

        network.info();

  /*      int countEpoch = 1000;
        for (int i = 0; i < countEpoch; i++) {
            for (int j = 0; j < 25; j++) {
                NNData3D data = loader.getNextTrainData(64);
                System.out.println(j + " - " + network.train(data.getOutput(), data.getInput()));
            }
            NNData3D test = loader.getNextTestData(1);
            NNTensor testQuery = NNArrays.isTensor(network.query(test.getOutput()))[0];

            ImageCreator.drawColorImage(testQuery, 8, 8, i + "_test", "D:/NetworkTest/flickr/", true);
            ImageCreator.drawColorImage(NNArrays.isTensor(test.getInput())[0], 8, 8, i + "_real", "D:/NetworkTest/flickr/", true);
            System.out.println("Validation " + network.accuracy(test.getInput()));
            network.save(new FileWriter("D:/NetworkTest/flickr/flick_network_8.txt"));
        }*/
    }
}
