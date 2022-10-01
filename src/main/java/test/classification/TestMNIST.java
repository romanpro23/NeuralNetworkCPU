package test.classification;

import data.mnist.MNISTLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.BatchRenormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNVector;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class TestMNIST {
    public static void main(String[] args) throws Exception {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(256))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDropoutLayer(0.7)
                .addLayer(new DenseLayer(128))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDropoutLayer(0.7)
                .addLayer(new DenseLayer(10).setTrainable(false))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/test.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader1D loader = new MNISTLoader1D();

        DataTrainer trainer = new DataTrainer(5000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
//            trainer.score(network, new DataMetric.Top1());
            trainer.train(network, 64, 1, new DataMetric.Top1());
//            network.save(new FileWriter(new File("test.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
