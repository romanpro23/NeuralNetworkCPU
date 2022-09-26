package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import data.network_train.NNData1D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.dense.VariationalLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.ConditionalVAE;
import neural_network.network.autoencoders.VariationalAutoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TestConditionalVAE {
    public static void main(String[] args) throws IOException {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(794)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new VariationalLayer(32))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(42)
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

        ConditionalVAE autoencoder = new ConditionalVAE(encoder, decoder);
        MNISTLoader1D loader = new MNISTLoader1D();
        Initializer initializer = new Initializer.RandomNormal();

        decoder.info();
        encoder.info();

        for (int i = 0; i < 100000; i++) {
            if (i % 50 == 0) {
                NNData1D data = loader.getNextTestData(1);
                ImageCreator.drawImage(NNArrays.isVector(data.getInput())[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/CVAE");
                NNVector result = NNArrays.toVector(autoencoder.query(data.getInput(), data.getOutput()))[0];
                ImageCreator.drawImage(result, 28, 28, i / 50 + "_output", "D:/NetworkTest/CVAE");
                NNVector resultVae = NNArrays.toVector(autoencoder.queryVariational(data.getInput(), data.getOutput()))[0];
                ImageCreator.drawImage(resultVae, 28, 28, i / 50 + "_output_vae", "D:/NetworkTest/CVAE");

                if(i % 250 == 0){
                    NNVector[] random = new NNVector[10];
                    NNVector[] dataN = new NNVector[10];
                    NNVector[] number = new NNVector[10];
                    NNVector rand = new NNVector(32);
                    initializer.initialize(rand);
                    for (int j = 0; j < 10; j++) {
                        random[j] = rand;
                        dataN[j] = NNArrays.isVector(data.getInput())[0];
                        number[j] = new NNVector(10);
                        number[j].set(j, 1);
                    }
                    NNVector[] resultRandom = NNArrays.toVector(autoencoder.queryDecoder(random, number));
                    NNVector[] resultDiffNumber = NNArrays.toVector(autoencoder.query(dataN, number));
                    for (int j = 0; j < 10; j++) {
                        ImageCreator.drawImage(resultRandom[j], 28, 28, i / 250 + "_" + j + "_random", "D:/NetworkTest/CVAE");
                        ImageCreator.drawImage(resultDiffNumber[j], 28, 28, i / 250 + "_" + j + "_number", "D:/NetworkTest/CVAE");
                    }
                }

//                encoder.save(new FileWriter(new File("D:/NetworkTest/CVAE/encoder_32h.txt")));
//                decoder.save(new FileWriter(new File("D:/NetworkTest/CVAE/decoder_32h.txt")));
            }
            NNData1D train = loader.getNextTrainData(64);
            System.out.println(i + " - " + autoencoder.train(train.getInput(), train.getOutput()));
        }
    }
}
