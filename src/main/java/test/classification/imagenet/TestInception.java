package test.classification.imagenet;

import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.Inception;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestInception {
    public static void main(String[] args) throws Exception {
        Initializer initializer = new Initializer.HeNormal();
        NeuralNetwork inceptionV3 = new Inception()
                .addInputLayer(64, 64, 3)
                .addConvolutionLayer(16, 3)
                .addConvolutionLayer(32, 3)
                .addMaxPoolingLayer(3, 2)
                .addConvolutionLayer(40, 1)
                .addConvolutionLayer(96, 3)
                .addMaxPoolingLayer(3, 2)
                .addInceptionA(16, 2)
                .addInceptionA(32, 2)
                .addInceptionA(32, 2)
                .addInceptionB()
                .addInceptionC(5, 128/2, 2)
                .addInceptionC(5, 160/2, 2)
                .addInceptionC(5, 160/2, 2)
                .addInceptionC(5, 192/2, 2)
                .addInceptionD(5, 2)
                .addInceptionE(2)
                .addInceptionE(2)
                .addGlobalAveragePoolingLayer()
                .addDropoutLayer(0.4)
                .addDenseLayer(200, new FunctionActivation.Softmax())
                .createInception()
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();

//        NeuralNetwork inceptionV3 = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg13.txt")))
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .create();

        TinyImageNetLoader3D loader = new TinyImageNetLoader3D(new TransformData.Tanh());
        inceptionV3.info();

        DataTrainer trainer = new DataTrainer(1500, 300, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            inceptionV3.save(new FileWriter("D:/NetworkTest/Imagenet/inceptionV3.txt"));
            trainer.train(inceptionV3, 60, 1, new DataMetric.Top1());
            trainer.score(inceptionV3, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }


}
