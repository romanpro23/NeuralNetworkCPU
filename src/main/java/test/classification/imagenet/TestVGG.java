package test.classification.imagenet;

import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class TestVGG {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork vgg16 = new VGG()
//                .addInputLayer(64, 64, 3)
//                .addConvolutionLayer(32, 3)
//                .addConvolutionLayer(32, 3)
//                .addMaxPoolingLayer()
//                .addConvolutionLayer(64, 3)
//                .addConvolutionLayer(64, 3)
//                .addMaxPoolingLayer()
//                .addConvolutionLayer(128, 3)
//                .addConvolutionLayer(128, 3)
//                .addConvolutionLayer(128, 3)
//                .addMaxPoolingLayer()
//                .addConvolutionLayer(256, 3)
//                .addConvolutionLayer(256, 3)
//                .addConvolutionLayer(256, 3)
//                .addMaxPoolingLayer()
//                .addConvolutionLayer(512, 3)
//                .addConvolutionLayer(512, 3)
//                .addConvolutionLayer(512, 3)
//                .addMaxPoolingLayer()
//                .addDenseLayer(1024, new FunctionActivation.ReLU())
//                .addDropoutLayer(0.25)
//                .addDenseLayer(1024, new FunctionActivation.ReLU())
//                .addDropoutLayer(0.25)
//                .addDenseLayer(200, new FunctionActivation.Softmax())
//                .createVGG()
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .create();

        Optimizer optimizer = new AdaBeliefOptimizer(0.0001);
        NeuralNetwork vgg16 = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .setStopGradient(23)
                .create();

        optimizer.read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16_optimizer.txt")));

        TinyImageNetLoader3D loader = new TinyImageNetLoader3D(new TransformData.Tanh()).useCrop().useReverse();
        vgg16.info();

        DataTrainer trainer = new DataTrainer(2500, 250, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            vgg16.save(new FileWriter("D:/NetworkTest/Imagenet/vgg16.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/Imagenet/vgg16_optimizer.txt"));
            trainer.train(vgg16, 60, 1, new DataMetric.Top1());
            trainer.score(vgg16, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
