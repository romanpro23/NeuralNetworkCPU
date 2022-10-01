package test.gan;

import data.ImageCreator;
import data.mnist.BatchMNIST;
import data.mnist.MNISTLoader1D;
import data.loaders.TransformData;
import neural_network.initialization.Initializer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestGAN {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork discriminator = new NeuralNetwork()
//                .addInputLayer(784)
//                .addLayer(new DenseLayer(1024))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DropoutLayer(0.3))
//                .addLayer(new DenseLayer(512))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DropoutLayer(0.3))
//                .addLayer(new DenseLayer(256))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DropoutLayer(0.3))
//                .addLayer(new DenseLayer(1))
//                .addActivationLayer(new FunctionActivation.Sigmoid())
//                .setOptimizer(new AdamOptimizer(0.0001))
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .create();

        NeuralNetwork discriminator = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/GAN/discriminator_mnist.txt")))
                .setOptimizer(new AdamOptimizer(0.0001))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();
//
//        NeuralNetwork generator = new NeuralNetwork()
//                .addInputLayer(32)
//                .addLayer(new DenseLayer(256))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DenseLayer(512))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DenseLayer(1024))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addDenseLayer(784, new FunctionActivation.Tanh())
//                .setOptimizer(new AdamOptimizer(0.0001))
//                .create();

        NeuralNetwork generator = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/GAN/generator_mnist.txt")))
                .setOptimizer(new AdamOptimizer(0.0001))
                .create();

        GAN gan = new GAN(generator, discriminator);
        MNISTLoader1D loader = new MNISTLoader1D(BatchMNIST.MNIST, new TransformData.Tanh());
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 50 == 0) {
                NNVector[] data = NNArrays.isVector(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/GAN", true);
                generator.save(new FileWriter(new File("D:/NetworkTest/GAN/generator_mnist.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/GAN/discriminator_mnist.txt")));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(32);
                initializer.initialize(random[0]);
                NNVector resultRandom = NNArrays.toVector(gan.query(random))[0];
                ImageCreator.drawImage(resultRandom, 28, 28, i / 50 + "_test", "D:/NetworkTest/GAN", true);
            }
            if (i % 500 == 0) {
                for (int j = 0; j < 10; j++) {
                    NNVector[] random = new NNVector[1];
                    random[0] = new NNVector(32);
                    initializer.initialize(random[0]);
                    NNVector resultRandom = NNArrays.toVector(gan.query(random))[0];
                    ImageCreator.drawImage(resultRandom, 28, 28, i / 500 + "_random_" + j, "D:/NetworkTest/GAN", true);
                }
            }
            System.out.println(i + " - " + gan.train(loader.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
