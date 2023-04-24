package test.classification.mnist;

import data.ImageCreator;
import data.mnist.MNISTLoader3D;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.capsule.*;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ReshapeLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.CapsNet;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;

public class TestCapsuleNetwork {
    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(64, 9, 1, 0))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new PrimaryCapsuleLayer(8, 32, 9, 2, 0))
                .addLayer(new CapsuleLayer(10, 16))
                .addLayer(new DigitCapsuleLayer())
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(10, 16)
                .addLayer(new FlattenLayer2D())
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDenseLayer(1024, new FunctionActivation.ReLU())
                .addDenseLayer(784, new FunctionActivation.Sigmoid())
                .addLayer(new ReshapeLayer3D(28, 28, 1))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(1000, 64, loader);
        network.info();
        decoder.info();

        CapsNet capsNet = new CapsNet(network, decoder);

        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();

            NNData3D test = loader.getNextTestData(1);
            NNTensor[] data = NNArrays.isTensor(test.getInput());
            ImageCreator.drawImage(data[0], 28, 28, i + "_input", "D:/NetworkTest/CapsNet");
            NNTensor[] decode_data = NNArrays.isTensor(capsNet.queryDecoder(test.getInput(), test.getOutput()));
            ImageCreator.drawImage(decode_data[0], 28, 28, i + "_decode_label", "D:/NetworkTest/CapsNet");
            NNTensor[] decode_data_ = NNArrays.isTensor(capsNet.queryDecoder(test.getInput()));
            ImageCreator.drawImage(decode_data_[0], 28, 28, i + "_decode", "D:/NetworkTest/CapsNet");

            for (int j = 0; j < 25; j++) {
                NNData3D trainData = loader.getNextTrainData(32);
                capsNet.train(trainData.getInput(), trainData.getOutput(), 0.001f);
                System.out.print(".");
            }
            System.out.println();

            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter("capsnet.txt"));
            decoder.save(new FileWriter("decoder.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
