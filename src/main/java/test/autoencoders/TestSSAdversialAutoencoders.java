package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import data.network_train.NNData1D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.LayersBlock;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.SSAdversarialAutoencoder;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;

public class TestSSAdversialAutoencoders {
    public static void main(String[] args) throws Exception {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(784)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(512)
                .addActivationLayer(new FunctionActivation.ReLU())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        LayersBlock classificationBlock = new LayersBlock()
                .addLayer(new DenseLayer(10))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()));

        LayersBlock styleBlock = new LayersBlock()
                .addLayer(new DenseLayer(32));

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(42)
                .addDenseLayer(512)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(784)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork discriminatorStyle = new NeuralNetwork()
                .addInputLayer(32)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        NeuralNetwork discriminatorLabel = new NeuralNetwork()
                .addInputLayer(10)
                .addDenseLayer(1024)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(1)
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        SSAdversarialAutoencoder autoencoder = new SSAdversarialAutoencoder(encoder, decoder, classificationBlock, styleBlock)
                .setDiscriminators(discriminatorLabel, discriminatorStyle)
                .setOptimizersEncoder(new AdamOptimizer(), new AdamOptimizer(), new AdamOptimizer());
        MNISTLoader1D loader = new MNISTLoader1D();
        Initializer initializer = new Initializer.RandomNormal();

        decoder.info();
        System.out.println();
        encoder.info();
        System.out.println();
        discriminatorStyle.info();
        System.out.println();
        discriminatorLabel.info();

        Optimizer optimizerClassification = new AdamOptimizer();
        NeuralNetwork classificator = encoder.copy()
                .addLayers(classificationBlock.getLayers())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizerClassification)
                .initialize(optimizerClassification);

        DataTrainer trainer = new DataTrainer(60000, 10000, loader);

        NNData1D trainClassification[] = new NNData1D[10];
        for (int i = 0; i < 10; i++) {
            trainClassification[i] = loader.getNextTrainData(64);
        }

        for (int i = 0; i < 100000; i++) {
            if (i % 50 == 0) {
                NNData1D data = loader.getNextTestData(1);

                ImageCreator.drawImage(NNArrays.isVector(data.getInput())[0], 28, 28, i / 50 + "_input", "D:/NetworkTest/SSAAE");
                NNVector result = NNArrays.toVector(autoencoder.query(data.getInput()))[0];
                ImageCreator.drawImage(result, 28, 28, i / 50 + "_output", "D:/NetworkTest/SSAAE");

                encoder.save(new FileWriter(new File("D:/NetworkTest/SSAAE/encoder.txt")));
                decoder.save(new FileWriter(new File("D:/NetworkTest/SSAAE/decoder.txt")));
                discriminatorLabel.save(new FileWriter(new File("D:/NetworkTest/SSAAE/discriminatorL.txt")));
                discriminatorStyle.save(new FileWriter(new File("D:/NetworkTest/SSAAE/discriminatorS.txt")));
                classificationBlock.write(new FileWriter(new File("D:/NetworkTest/SSAAE/classificationBlock.txt")));
                styleBlock.write(new FileWriter(new File("D:/NetworkTest/SSAAE/styleBlock.txt")));

                if(i % 500 == 0){
                    NNVector[] random = new NNVector[10];
                    NNVector[] number = new NNVector[10];
                    for (int j = 0; j < 10; j++) {
                        NNVector rand = new NNVector(32);
                        initializer.initialize(rand);
                        random[j] = rand;
                        number[j] = new NNVector(10);
                        number[j].set(j, 1);
                    }
                    NNVector[] resultRandom = NNArrays.toVector(autoencoder.queryDecoder(number, random));
                    for (int j = 0; j < 10; j++) {
                        ImageCreator.drawImage(resultRandom[j], 28, 28, i / 500 + "_" + j + "_random", "D:/NetworkTest/SSAAE");
                    }
                }
            }
            NNData1D train = loader.getNextTrainData(64);
            System.out.println(i + " - " + autoencoder.train(train.getInput(), train.getOutput()));
            classificator.train(trainClassification[i%10].getInput(), trainClassification[i%10].getOutput());
            if(i % 200 == 0) {
                trainer.score(classificator, new DataMetric.Top1());
            }
        }
    }
}
