package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.ciraf.Ciraf10Loader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.DropoutLayer3D;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class TestVGG {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork vgg = new NeuralNetwork()
//                .addInputLayer(32, 32, 3)
//                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new Flatten3DLayer())
//                .addDenseLayer(512)
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new BatchNormalizationLayer(0.9))
//                .addDenseLayer(512)
//                .addLayer(new BatchNormalizationLayer(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addDenseLayer(100, new FunctionActivation.Softmax())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .create();
//
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg7_norm.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();
//
        vgg.info();

        Ciraf100Loader3D loader = new Ciraf100Loader3D();

        DataTrainer trainer = new DataTrainer(5000, 1000, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            vgg.save(new FileWriter("D:/NetworkTest/ciraf/vgg7_norm.txt"));
            trainer.train(vgg, 64, 1, new DataMetric.Top1());
            trainer.score(vgg, new DataMetric.Top1());
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
