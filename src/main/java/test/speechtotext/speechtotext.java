package test.speechtotext;

import data.nlp.PositionUaLoader;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;
import utilities.GPUInit;
import utilities.Use;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class speechtotext {
    static NeuralNetwork network;

    public static void main(String[] args) throws Exception {
        GPUInit.startup();

        PositionLoader loader = new PositionLoader(45);
        loader.setUseReverse(true);

        Optimizer optimizer = new AdamOptimizer();
        /*network = NeuralNetwork.read(new Scanner(new File("C:/Levani/NeuralNetwork/data/ka_speech_recognation.txt")))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.MSE())
                .setTrainable(true)
                .create();*/

                network = new NeuralNetwork();
                network.addInputLayer(24, 488, 1)
                //.addLayer(new ConvolutionLayer(64, 20, 42, 4, 0, 0))
                //.addLayer(new ConvolutionLayer(4096, 24, 492, 2, 0, 0))
                //.addLayer(new ConvolutionLayer(64, 20, 42, 4, 0, 0))
                //.addLayer(new FlattenLayer3D())
                //.addLayer(new ReshapeLayer2D(64, 64))
                .addLayer(new ImagePatchesLayer(8,64))
                .addLayer(new VITPositionalEmbeddingLayer())
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64).setMask())
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                        .addLayer(new DenseLayer2D(64))
                        .addLayer(new DropoutLayer2D(0.00001))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64).setMask())
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                        .addLayer(new DenseLayer2D(64))
                        .addLayer(new DropoutLayer2D(0.00001))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(4, 64).setMask())
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(128))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                        .addLayer(new DenseLayer2D(64))
                        .addLayer(new DropoutLayer2D(0.00001))
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new FlattenLayer2D())
                .addLayer(new DenseLayer(175))
                .addLayer(new ActivationLayer(new FunctionActivation.Linear()))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.MSE())
                .setTrainable(true)
                .create();

  //      optimizer.read(new Scanner(new File("C:/Levani/NeuralNetwork/data/ka_speech_recognation_optimizer.txt")));

        network.info();
        DataTrainer trainer = new DataTrainer(5000, 5000, loader);

        for (int i = 0; i < 1000; i++) {
            long start = System.nanoTime();
            trainer.train(network, 8, 1, new DataMetric.Top1());

  //          network.save(new FileWriter("C:/Levani/NeuralNetwork/data/ka_speech_recognation.txt"));
  //          optimizer.save(new FileWriter("C:/Levani/NeuralNetwork/data/ka_speech_recognation_optimizer.txt"));

            System.out.println((System.nanoTime() - start) / 1000000);

            for (int s = 0; s < 10; s++) {
                float[] result = network.query(loader.getNextTestData(1).getInput())[0].getData();
                String Text = loader.decodeString(result);

                byte[] charset = Text.getBytes("UTF-8");
                String newstr = new String(charset, "UTF-8");
                System.out.println(newstr);
            }
        }
    }
}

