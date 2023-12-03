package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;

public class TestVIT {
    public static void main(String[] args) throws IOException {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ImagePatchesLayer(4, 64))
                .addLayer(new VITPositionalEmbeddingLayer(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64, false).setMask())
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128, false))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
                        .addLayer(new DenseLayer2D(64, false))
                        .addLayer(new DropoutLayer2D(0.2))
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64, false).setMask())
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128, false))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
                        .addLayer(new DenseLayer2D(64, false))
                        .addLayer(new DropoutLayer2D(0.2))
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64, false).setMask())
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128, false))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
                        .addLayer(new DenseLayer2D(64, false))
                        .addLayer(new DropoutLayer2D(0.2))
                )
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new FlattenLayer2D(false))
                .addLayer(new DenseLayer(10, false))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(10000, 10000, loader);
        network.info();

//        trainer.score(network, new DataMetric.Top1());
        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            trainer.train(network, 64, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
            network.save(new FileWriter("vit_gelu.txt"));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
