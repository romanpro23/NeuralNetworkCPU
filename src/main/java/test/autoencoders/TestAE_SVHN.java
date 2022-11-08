package test.autoencoders;

import data.ImageCreator;
import data.loaders.TransformData;
import data.svhn.SVHNLoader3D;
import neural_network.initialization.Initializer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.VariationalAutoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestAE_SVHN {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork encoder = new NeuralNetwork()
//                .addInputLayer(32, 32, 3)
//                .addLayer(new ConvolutionLayer(32, 4, 2, 1))
//                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
//                .addLayer(new ConvolutionLayer(64, 4, 2, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
//                .addLayer(new ConvolutionLayer(128, 4, 2, 1))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
//                .addLayer(new Flatten3DLayer())
//                .addLayer(new VariationalLayer(100))
//                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
//                .create();
//
//        NeuralNetwork decoder = new NeuralNetwork()
//                .addInputLayer(100)
//                .addLayer(new DenseLayer(2048).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addLayer(new Reshape3DLayer(4, 4, 128))
//                .addLayer(new ConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addActivationLayer(new FunctionActivation.Tanh())
//                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .create();

        NeuralNetwork encoder = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/svhn/VAE/encoder_svhn.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
                .create();

        NeuralNetwork decoder = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/svhn/VAE/decoder_svhn.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        VariationalAutoencoder gan = new VariationalAutoencoder(encoder, decoder);
        SVHNLoader3D loader = new SVHNLoader3D(new TransformData.Tanh());
        Initializer initializer = new Initializer.RandomNormal();

        decoder.info();
        encoder.info();

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 32, 32, i / 10 + "_input", "D:/NetworkTest/svhn/VAE", true);
                decoder.save(new FileWriter(new File("D:/NetworkTest/svhn/VAE/decoder_svhn.txt")));
                encoder.save(new FileWriter(new File("D:/NetworkTest/svhn/VAE/encoder_svhn.txt")));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNTensor resultRandom = NNArrays.isTensor(gan.queryDecoder(random))[0];
                ImageCreator.drawColorImage(resultRandom, 32, 32, i /10 + "_rand", "D:/NetworkTest/svhn/VAE", true);
                NNTensor resultDecode = NNArrays.isTensor(gan.query(data))[0];
                ImageCreator.drawColorImage(resultDecode, 32, 32, i /10 + "_test", "D:/NetworkTest/svhn/VAE", true);
            }
            System.out.println(i + " - " + gan.train(loader.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
