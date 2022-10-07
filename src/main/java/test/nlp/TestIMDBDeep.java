package test.nlp;

import data.imdb.IMDBLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_2d.ConvolutionLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.layers.recurrent.GRULayer;
import neural_network.layers.recurrent.RecurrentLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.layers.reshape.Flatten2DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

public class TestIMDBDeep {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(5000, 32).setTrainable(true))
                .addLayer(new GRULayer(64))
                .addLayer(new Flatten2DLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(64).setTrainable(true))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DropoutLayer(0.3))
                .addLayer(new DenseLayer(1).setTrainable(true))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        network.info();

        IMDBLoader1D loader = new IMDBLoader1D();
        DataTrainer trainer = new DataTrainer(25000, 25000, loader);

        for (int i = 0; i < 10; i++) {
            trainer.train(network, 128, 1, new DataMetric.Binary());
        }
    }
}
