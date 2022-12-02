package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestMNIST3D {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new AdamOptimizer();
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(8, 3, 1, 1).setTrainable(true))
                .addLayer(new ParametricReLULayer3D())
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(16, 3, 1, 1).setTrainable(true))
                .addLayer(new ParametricReLULayer3D())
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new Flatten3DLayer())
                .addLayer(new DenseLayer(256).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer(10).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/testA.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(1000, 1000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            trainer.train(network, 64, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter(new File("test.txt")));
            optimizer.save(new FileWriter(new File("testOptimizer.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
