package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.capsule.CapsuleLayer;
import neural_network.layers.capsule.DigitCapsuleLayer;
import neural_network.layers.capsule.PrimaryCapsuleLayer;
import neural_network.layers.capsule.SquashActivationLayer;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_2d.DenseLayer2D;
import neural_network.layers.layer_2d.VITPositionalEmbeddingLayer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class TestCapsule {
    public static void main(String[] args) throws IOException {
        CapsuleLayer cl;
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(32, 3, 2, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new PrimaryCapsuleLayer(8, 8, 5, 2, 1))
                .addLayer(new SquashActivationLayer())
                .addLayer(cl = new CapsuleLayer(64, 12))
                .addLayer(cl = new CapsuleLayer(10, 16))
                .addLayer(new DigitCapsuleLayer())
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Capsule())
                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(20000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

            NNData3D data = loader.getNextTestData(1);
            System.out.println(Arrays.toString(network.query(data.getInput())[0].getData()));
            System.out.println(Arrays.toString(data.getOutput()[0].getData()));

            trainer.train(network, 32, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter("capsNet_3.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
