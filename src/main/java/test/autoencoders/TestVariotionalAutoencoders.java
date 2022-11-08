package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.dense.VariationalLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.VariationalAutoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

public class TestVariotionalAutoencoders {
    public static void main(String[] args) throws Exception {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(784)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new VariationalLayer(8))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(8)
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(784)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

//        NeuralNetwork encoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/VAE/encoder_32h.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .create();
//
//        NeuralNetwork decoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/VAE/decoder_32h.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .create();

        VariationalAutoencoder autoencoder = new VariationalAutoencoder(encoder, decoder);
        MNISTLoader1D loader = new MNISTLoader1D();
        Initializer initializer = new Initializer.RandomNormal();

        decoder.info();
        encoder.info();

        for (int i = 0; i < 100000; i++) {
            if (i % 50 == 0) {
                NNVector[] data = NNArrays.isVector(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/VAE");
                NNVector result = NNArrays.toVector(autoencoder.query(data))[0];
                ImageCreator.drawImage(result, 28, 28, i / 50 + "_output", "D:/NetworkTest/VAE");
                NNVector resultVae = NNArrays.toVector(autoencoder.queryVariational(data))[0];
                ImageCreator.drawImage(resultVae, 28, 28, i / 50 + "_output_vae", "D:/NetworkTest/VAE");
                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(32);
                initializer.initialize(random[0]);
                NNVector resultRandom = NNArrays.toVector(autoencoder.queryDecoder(random))[0];
                ImageCreator.drawImage(resultRandom, 28, 28, i / 50 + "_random", "D:/NetworkTest/VAE");

//                encoder.save(new FileWriter(new File("D:/NetworkTest/VAE/encoder_32h.txt")));
//                decoder.save(new FileWriter(new File("D:/NetworkTest/VAE/decoder_32h.txt")));
            }
            System.out.println(i + " - " + autoencoder.train(loader.getNextTrainData(64).getInput()));
        }
    }
}
