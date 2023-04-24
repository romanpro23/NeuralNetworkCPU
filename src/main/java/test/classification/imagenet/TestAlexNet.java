package test.classification.imagenet;

import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.AlexNet;
import neural_network.optimizers.*;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;

public class TestAlexNet {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new AdaBeliefOptimizer();

        NeuralNetwork alexnet = new AlexNet()
                .addInputLayer(64, 64, 3)
                .addConvolutionLayer(48, 7, 2, 3, true)
                .addMaxPoolingLayer(3, 2, 1)
                .addConvolutionLayer(128, 5, 1, 2, true)
                .addMaxPoolingLayer(3, 2, 1)
                .addConvolutionLayer(196, 3, 1, 1, true)
                .addConvolutionLayer(196, 3, 1, 1, true)
                .addConvolutionLayer(128, 3, 1, 1, true)
                .addMaxPoolingLayer(3, 2, 1)
                .addDenseLayer(1024, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(1024, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(250, new FunctionActivation.Softmax())
                .createAlexNet()
                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
                .setOptimizer(optimizer)
                .create();
//
//        NeuralNetwork alexnet = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/Imagenet/_alexnet.txt")))
//                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
//                .setTrainable(true)
//                .addOptimizer(optimizer,9)
//                .addOptimizer(optimizerConv,0,9)
//                .create();

//        optimizer.read(new Scanner(new File("D:/NetworkTest/Imagenet/_alexnet_optimizer.txt")));
//        optimizerConv.read(new Scanner(new File("D:/NetworkTest/Imagenet/_alexnet_conv_optimizer.txt")));

        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useReverse().useCrop();
        alexnet.info();

        DataTrainer trainer = new DataTrainer(5120/2, 512/2, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            alexnet.save(new FileWriter("D:/NetworkTest/Imagenet/alexnet_norm.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/Imagenet/alexnet_optimizer_norm.txt"));
            trainer.train(alexnet, 128, 1, new DataMetric.Top1());
            if (i % 2 == 0) {
                trainer.score(alexnet, new DataMetric.Top1());
            } else {
                trainer.score(alexnet, new DataMetric.Top5());
            }

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}