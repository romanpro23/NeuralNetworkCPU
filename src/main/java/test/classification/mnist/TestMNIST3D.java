package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

public class TestMNIST3D {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new AdamOptimizer(0.1);
        boolean trainable = true;
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(8, 3, 1, 1).setTrainable(trainable))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(16, 3, 1, 1).setTrainable(trainable))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(32, 3, 1, 1).setTrainable(trainable))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new FlattenLayer3D())
//                .addLayer(new DenseLayer(256).setTrainable(false))
//                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer(10, false).setTrainable(false))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/testA.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(10000, 10000, loader);
        network.info();

//        trainer.score(network, new DataMetric.Top1());
        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            trainer.train(network, 64, 1, new DataMetric.Top1());
            trainer.score(network, new DataMetric.Top1());
//            network.save(new FileWriter(new File("modulatedV2_conv.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
