package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestVGG {
    public static void main(String[] args) throws Exception {
        Optimizer optimizerConv = new AdaBeliefOptimizer(0.0001);
        Optimizer optimizer = new AdaBeliefOptimizer(0.0001);
        NeuralNetwork vgg16 = new VGG()
                .addInputLayer(32, 32, 3)
                .addConvolutionLayer(32, 3)
                .addConvolutionLayer(32, 3)
                .addMaxPoolingLayer()
                .addConvolutionLayer(64, 3)
                .addConvolutionLayer(64, 3)
                .addMaxPoolingLayer()
                .addConvolutionLayer(128, 3)
                .addConvolutionLayer(128, 3)
                .addConvolutionLayer(128, 3)
                .addMaxPoolingLayer()
                .addConvolutionLayer(256, 3)
                .addConvolutionLayer(256, 3)
                .addConvolutionLayer(256, 3)
                .addMaxPoolingLayer()
                .addConvolutionLayer(512, 3)
                .addConvolutionLayer(512, 3)
                .addConvolutionLayer(512, 3)
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(100, new FunctionActivation.Softmax())
                .createVGG()
                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
//                .setOptimizer(optimizer)
                .setStopGradient(30)
                .addOptimizer(optimizer, 30)
//                .addOptimizer(optimizerConv, 0, 30)
                .create();

//        NeuralNetwork vgg16 = NeuralNetwork.read("D:/NetworkTest/ciraf/_vgg16.txt")
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setStopGradient(28)
//                .addOptimizer(optimizer, 28)
//                .addOptimizer(optimizerConv, 0, 30)
//                .create();
//
//        optimizer.read("D:/NetworkTest/ciraf/_vgg16_dense_optimizer.txt");
//        optimizerConv.read("D:/NetworkTest/ciraf/vgg16_conv_optimizer.txt");

        vgg16.info();

        Ciraf100Loader3D loader = new Ciraf100Loader3D(new TransformData.Tanh()).useReverse();

        DataTrainer trainer = new DataTrainer(5120, 512, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

            vgg16.save(new FileWriter("D:/NetworkTest/ciraf/_vgg16.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/ciraf/_vgg16_dense_optimizer.txt"));
//            optimizerConv.save(new FileWriter("D:/NetworkTest/ciraf/vgg16_conv_optimizer.txt"));

            trainer.train(vgg16, 64, 1, 1, new DataMetric.Top1());
            trainer.score(vgg16, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
