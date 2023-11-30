package test.nlp;

import data.imdb.IMDBLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_2d.ActivationLayer2D;
import neural_network.layers.layer_2d.AdditionBlock;
import neural_network.layers.layer_2d.DropoutLayer2D;
import neural_network.layers.layer_2d.NormalizationLayer2D;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.DenseLayer2D;
import neural_network.layers.layer_2d.MultiHeadAttentionLayer;
import neural_network.layers.layer_2d.PositionalEmbeddingLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

public class TestIMDBTransformer {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(5000, 64).setTrainable(true))
                .addLayer(new PositionalEmbeddingLayer())
                .addLayer(new AdditionBlock()
                    .addLayer(new MultiHeadAttentionLayer(4, 64, false).setMask())
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128, false))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.ReLU()))
                        .addLayer(new DenseLayer2D(64, false))
                        .addLayer(new DropoutLayer2D(0.2))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                    .addLayer(new MultiHeadAttentionLayer(4, 64, false))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128, false))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.ReLU()))
                        .addLayer(new DenseLayer2D(64, false))
                        .addLayer(new DropoutLayer2D(0.2))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new FlattenLayer2D())
                .addLayer(new DenseLayer(1).setTrainable(true))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        network.info();

        IMDBLoader1D loader = new IMDBLoader1D();
        DataTrainer trainer = new DataTrainer(10000, 10000, loader);

        for (int i = 0; i < 10; i++) {
            trainer.train(network, 100, 1, new DataMetric.Binary());
        }
    }
}
