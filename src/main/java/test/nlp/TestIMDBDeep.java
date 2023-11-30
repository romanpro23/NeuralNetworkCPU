package test.nlp;

import data.imdb.IMDBLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_2d.MultiHeadAttentionLayer;
import neural_network.layers.layer_2d.NormalizationLayer2D;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.PositionalEmbeddingLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

public class TestIMDBDeep {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(5000, 64).setTrainable(false))
                .addLayer(new PositionalEmbeddingLayer())
                .addLayer(new MultiHeadAttentionLayer(1, 64, false).setTrainable(true).setMask())
                .addLayer(new NormalizationLayer2D().setTrainable(false))
                .addLayer(new FlattenLayer2D())
                .addLayer(new DenseLayer(1).setTrainable(false))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        network.info();

        IMDBLoader1D loader = new IMDBLoader1D();
        DataTrainer trainer = new DataTrainer(5000, 5000, loader);

        for (int i = 0; i < 10; i++) {
            trainer.train(network, 1, 1, new DataMetric.Binary());
        }
    }
}
