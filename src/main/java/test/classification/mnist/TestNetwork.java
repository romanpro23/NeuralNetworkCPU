package test.classification.mnist;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionTransposeLayer;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.reshape.ReshapeLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;
import nnarrays.NNVector;

public class TestNetwork {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(32)
                .addLayer(new DenseLayer(14 * 14 * 4, false).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new ReshapeLayer3D(7, 7, 16))
                .addLayer(new ConvolutionTransposeLayer(8, 4, 2, 1).setTrainable(false))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setTrainable(false))
                .addLayer(new ActivationLayer3D(new FunctionActivation.Sigmoid()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NNVector[] input = new NNVector[64];
        NNTensor[] output = new NNTensor[input.length];

        int inputSize = 32;
        int outputSize = 784;

        network.info();
        Initializer initializer = new Initializer.RandomNormal();

        for (int i = 0; i < input.length; i++) {
            input[i] = new NNVector(inputSize);
            output[i] = new NNTensor(28, 28, 3);

            initializer.initialize(input[i]);
            initializer.initialize(output[i]);
            output[i].sigmoid(output[i]);
        }

        for (int i = 0; i < 128; i++) {
            /*long start = System.nanoTime();
            System.out.println(network.train(input, output) / (64 * 784 * 3));
            System.out.println((System.nanoTime() - start) / 1000000);*/
        }
    }
}
