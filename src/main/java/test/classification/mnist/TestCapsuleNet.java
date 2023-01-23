package test.classification.mnist;

import data.ImageCreator;
import data.loaders.TransformData;
import data.mnist.BatchMNIST;
import data.mnist.MNISTLoader3D;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.capsule.CapsuleLayer;
import neural_network.layers.capsule.DigitCapsuleLayer;
import neural_network.layers.capsule.PrimaryCapsuleLayer;
import neural_network.layers.capsule.SquashActivationLayer;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.reshape.Flatten2DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
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

public class TestCapsuleNet {
    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(8, 9, 1, 0).setTrainable(true))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new PrimaryCapsuleLayer(8, 16, 9, 2, 0).setTrainable(true))
                .addLayer(new SquashActivationLayer())
                .addLayer(new CapsuleLayer(10, 16).setTrainable(true))
                .addLayer(new DigitCapsuleLayer())
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Capsule())
                .create();

//        NeuralNetwork decoder = new NeuralNetwork()
//                .addInputLayer(10, 16)
//                .addLayer(new Flatten2DLayer())
//                .addDenseLayer(512, new FunctionActivation.ReLU())
//                .addDenseLayer(1024, new FunctionActivation.ReLU())
//                .addDenseLayer(784, new FunctionActivation.Tanh())
//                .addLayer(new Reshape3DLayer(28, 28, 1))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(1000, 1000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

            trainer.train(network, 32, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter("capsnet.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
