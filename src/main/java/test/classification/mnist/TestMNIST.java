package test.classification.mnist;

import data.mnist.MNISTLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.SGDOptimizer;
import neural_network.regularization.Regularization;
import trainer.DataMetric;
import trainer.DataTrainer;

public class TestMNIST {
    public static void main(String[] args) throws Exception {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(64))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(64))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(10))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/testA.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader1D loader = new MNISTLoader1D();

        DataTrainer trainer = new DataTrainer(60000, 10000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
//            trainer.score(network, new DataMetric.Top1());
            trainer.train(network, 128, 1, new DataMetric.Top1());
//            network.save(new FileWriter(new File("testA.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
