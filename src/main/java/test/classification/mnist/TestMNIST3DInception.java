package test.classification.mnist;

import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.convolution_3d.inception.InceptionBlock;
import neural_network.layers.convolution_3d.inception.InceptionModule;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.GlobalAveragePooling3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;

public class TestMNIST3DInception {
    public static void main(String[] args) throws Exception {
        NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(8, 3, 2, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new InceptionModule()
                        .addInceptionBlock(new InceptionBlock()
                                .addLayer(new ConvolutionLayer(4, 3, 1, 1))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionBlock(new InceptionBlock()
                                .addLayer(new ConvolutionLayer(4, 5, 1, 2))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionBlock(new InceptionBlock()
                                .addLayer(new ConvolutionLayer(4, 1, 1, 0))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new MaxPoolingLayer(3, 1, 1))
                        )
                )
                .addLayer(new ConvolutionLayer(16, 3, 2, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new Flatten3DLayer())
                .addLayer(new DenseLayer(128).setTrainable(true))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(10).setTrainable(true))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

//        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("D:/testA.txt")))
//                .setOptimizer(new AdamOptimizer())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .create();

        MNISTLoader3D loader = new MNISTLoader3D();

        DataTrainer trainer = new DataTrainer(1000, 1000, loader);
        network.info();

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
//            trainer.score(network, new DataMetric.Top1());
            trainer.train(network, 64, 1, new DataMetric.Top1());
            network.save(new FileWriter(new File("testA.txt")));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
