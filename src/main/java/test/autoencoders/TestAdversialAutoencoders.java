package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.AdversarialAutoencoder;
import neural_network.network.autoencoders.VariationalAutoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestAdversialAutoencoders {
    public static void main(String[] args) throws Exception {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(784)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(32))
                .setOptimizer(new AdamOptimizer(0.0006))
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(32)
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(784)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.0006))
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(32)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(512)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.0008))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        AdversarialAutoencoder autoencoder = new AdversarialAutoencoder(encoder, decoder, discriminator);
        MNISTLoader1D loader = new MNISTLoader1D();
        Initializer initializer = new Initializer.RandomNormal();

        decoder.info();
        encoder.info();
        discriminator.info();
        autoencoder.setOptimizersEncoder(new AdamOptimizer(0.0006), new AdamOptimizer(0.0008));

        for (int i = 0; i < 100000; i++) {
            if (i % 50 == 0) {
                NNVector[] data = NNArrays.isVector(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/AAE");
                NNVector result = NNArrays.toVector(autoencoder.query(data))[0];
                ImageCreator.drawImage(result, 28, 28, i / 50 + "_output", "D:/NetworkTest/AAE");
                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(32);
                initializer.initialize(random[0]);
                NNVector resultRandom = NNArrays.toVector(autoencoder.queryDecoder(random))[0];
                ImageCreator.drawImage(resultRandom, 28, 28, i / 50 + "_random", "D:/NetworkTest/AAE");

                encoder.save(new FileWriter(new File("D:/NetworkTest/AAE/encoder_32h.txt")));
                decoder.save(new FileWriter(new File("D:/NetworkTest/AAE/decoder_32h.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/AAE/discriminator_32h.txt")));
            }
            System.out.println(i + " - " + autoencoder.train(loader.getNextTrainData(64).getInput()));
        }
    }
}
