package test.classification.imagenet;

import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.AdaBeliefOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestVGG16 {
    public static void main(String[] args) throws Exception {
        AdaBeliefOptimizer optimizer = new AdaBeliefOptimizer(0.0001);
//        AdaBeliefOptimizer optimizerConv = new AdaBeliefOptimizer(0.0001);

        NeuralNetwork vgg16 = new VGG()
                .addInputLayer(64, 64, 3)
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
                .addMaxPoolingLayer()
                .addDenseLayer(2048, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(2048, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(250, new FunctionActivation.Softmax())
                .createVGG()
                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
                .setOptimizer(optimizer)
//                .addOptimizer(optimizerConv, 0, 30)
                .create();

//        NeuralNetwork vgg16 = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/Imagenet/_vgg16.txt")))
//                .setTrainable(true)
//                .setOptimizer(optimizerDense)
//                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
//                .create();

        vgg16.info();
//        optimizerDense.read(new Scanner(new File("D:/NetworkTest/Imagenet/_vgg16_optimizer.txt")));

        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useReverse().useCrop();
        DataTrainer trainer = new DataTrainer(5120 / 4, 512 / 4, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            vgg16.save(new FileWriter("D:/NetworkTest/Imagenet/vgg16.txt"));
//            optimizerConv.save(new FileWriter("D:/NetworkTest/Imagenet/vgg16_conv_optimizer.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/Imagenet/vgg16_optimizer.txt"));
            trainer.train(vgg16, 32, 1, 4, new DataMetric.Top1());

            if (i % 2 == 0) {
                trainer.score(vgg16, new DataMetric.Top1());
            } else {
                trainer.score(vgg16, new DataMetric.Top5());
            }

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}