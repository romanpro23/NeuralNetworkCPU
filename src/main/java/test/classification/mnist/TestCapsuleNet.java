package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.capsule.*;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ReshapeLayer2D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

public class TestCapsuleNet {
    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(10, 9, 1, 0).setTrainable(true))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new PrimaryCapsuleLayer(8, 8, 9, 2, 0).setTrainable(true))
                .addLayer(new CapsuleLayer(10, 16).setTrainable(true))
                .addLayer(new DigitCapsuleLayer(false))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
//                .setStopGradient(3)
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Capsule())
                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(10000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

            trainer.train(network, 128, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());

            NNData3D data = loader.getNextTestData(1);
            System.out.println(Arrays.toString(network.query(data.getInput())[0].getData()));
            System.out.println(Arrays.toString(data.getOutput()[0].getData()));
            network.save(new FileWriter("capsnet_10.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
