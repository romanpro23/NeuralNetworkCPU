package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.Autoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TestAutoencoders {
    public static void main(String[] args) throws IOException {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(784)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(32))
                .setOptimizer(new AdamOptimizer())
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
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        Autoencoder autoencoder = new Autoencoder(encoder, decoder);
        MNISTLoader1D loader = new MNISTLoader1D();

        for (int i = 0; i < 100000; i++) {
            if (i % 50 == 0) {
                NNVector[] data = NNArrays.isVector(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/AE");
                NNVector result = NNArrays.toVector(autoencoder.query(data))[0];
                ImageCreator.drawImage(result, 28, 28, i / 50 + "_output", "D:/NetworkTest/AE");
                encoder.save(new FileWriter(new File("D:/NetworkTest/AE/encoder_32h.txt")));
                decoder.save(new FileWriter(new File("D:/NetworkTest/AE/decoder_32h.txt")));
            }
            System.out.println(i + " - " + autoencoder.train(loader.getNextTrainData(64).getInput()));
        }
    }
}
