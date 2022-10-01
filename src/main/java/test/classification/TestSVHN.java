package test.classification;

import data.ImageCreator;
import data.ciraf.Ciraf100Loader3D;
import data.ciraf.Ciraf10Loader3D;
import data.mnist.MNISTLoader1D;
import data.svhn.SVHNLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.BatchRenormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;

public class TestSVHN {
    public static void main(String[] args) throws IOException {
        SVHNLoader1D loader = new SVHNLoader1D();

        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(3072)
                .addLayer(new DenseLayer(2056))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(512))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(256))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(10))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

        DataTrainer trainer = new DataTrainer(10000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            trainer.train(network, 64, 1, new DataMetric.Top1());
            network.save(new FileWriter("D:/NetworkTest/svhn/deeep_network.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
