package test;

import neural_network.activation.FunctionActivation;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNVector;

public class TestNetwork {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(512))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(10))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        NNVector[] input = new NNVector[128];
        NNVector[] output = new NNVector[input.length];

        int inputSize = 784;
        int outputSize = 64;

        for (int i = 0; i < input.length; i++) {
            input[i] = new NNVector(inputSize);
            output[i] = new NNVector(outputSize);

            for (int j = 0; j < inputSize; j++) {
                input[i].set(j, (float) (Math.random() - 0.5f));
            }
            for (int j = 0; j < outputSize; j++) {
                output[i].set(j, (float) (Math.random()));
            }
        }

        for (int i = 0; i < 40960; i++) {
            long start = System.nanoTime();
            network.train(input, output);
            System.out.println(network.accuracy(output));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
