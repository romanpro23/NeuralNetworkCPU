package test.classification.mnist;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.GlobalAveragePooling3DLayer;
import neural_network.layers.reshape.GlobalMaxPooling3DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;

public class TestMNIST3D {
    public static void main(String[] args) throws Exception {
        SelfAttentionLayer l1, l2;
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(8, 3, 1, 1).setTrainable(true))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(l1 = new SelfAttentionLayer())
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(16, 3, 1, 1).setTrainable(true))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(l2 = new SelfAttentionLayer())
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new Flatten3DLayer())
                .addLayer(new DenseLayer(256).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer(10).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/testA.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(10000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            trainer.train(network, 64, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter(new File("D:/NetworkTest/SAGAN/mnistAtt.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);

            NNTensor input = (NNTensor) loader.getNextTestData(1).getInput()[0];

            ImageCreator.drawImage(input, 28, 28, i + "_input", "D:/NetworkTest/SAGAN", false);

//            NNMatrix attention1 = l1.getAttention()[0].transpose();
//            attention1.sigmoid(attention1);
//            ImageCreator.drawImage(new NNTensor(28, 28, 28*28, attention1.getData()), 28, 28, i + "_attention1", "D:/NetworkTest/SAGAN", false);
            NNMatrix attention2 = l2.getAttention()[0].transpose();
            ImageCreator.drawImage(new NNTensor(14*14, 14*14, 1, attention2.getData()), 14*14, 14*14, i + "_attention2", "D:/NetworkTest/SAGAN", false);
        }
    }
}
