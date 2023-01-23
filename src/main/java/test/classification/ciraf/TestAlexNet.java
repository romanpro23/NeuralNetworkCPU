package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.imageNet.ImageNet1kLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.AlexNet;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.MomentumOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestAlexNet {
    public static void main(String[] args) throws Exception {
       Optimizer optimizer = new AdaBeliefOptimizer(0.001);
//        AdaBeliefOptimizer optimizerConv = new AdaBeliefOptimizer(0.001);

        NeuralNetwork alexnet = new AlexNet()
                .addInputLayer(32, 32, 3)
                .addConvolutionLayer(48, 7, 2, 3)
                .addMaxPoolingLayer(3, 2, 1)
                .addConvolutionLayer(128, 5, 1, 2)
                .addMaxPoolingLayer(3, 2, 1)
                .addConvolutionLayer(196, 3, 1, 1)
                .addConvolutionLayer(196, 3, 1, 1)
                .addConvolutionLayer(128, 3, 1, 1)
                .addMaxPoolingLayer(3, 2, 1)
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(100, new FunctionActivation.Softmax())
                .createAlexNet()
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .setStopGradient(12)
//                .setOptimizerConv(optimizerConv)
                .create();

//        NeuralNetwork alexnet = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/ciraf/alexnet.txt")))
//                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
//                .setOptimizerConv(optimizerConv)
//                .setOptimizer(optimizer)
////                .setStopGradient(13)
//                .create();
//
//        optimizer.read(new Scanner(new File("D:/NetworkTest/ciraf/alexnet_optimizer_dense.txt")));
//        optimizerConv.read(new Scanner(new File("D:/NetworkTest/ciraf/alexnet_optimizer_conv.txt")));

        Ciraf100Loader3D loader = new Ciraf100Loader3D(new TransformData.Tanh()).useReverse();
        alexnet.info();

        DataTrainer trainer = new DataTrainer(5120/4, 512/4, loader);

        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            alexnet.save(new FileWriter("D:/NetworkTest/ciraf/_alexnet.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/ciraf/alexnet_optimizer_dense.txt"));
//            optimizerConv.save(new FileWriter("D:/NetworkTest/ciraf/alexnet_optimizer_conv.txt"));
            trainer.train(alexnet, 128, 1, new DataMetric.Top1());
            if (i % 2 == 0) {
                trainer.score(alexnet, new DataMetric.Top1());
            } else {
                trainer.score(alexnet, new DataMetric.Top5());
            }
//            if(i == 100){
//                optimizer.setLearningRate(0.0001f);
//            }
//            if(i == 200){
//                optimizer.setLearningRate(0.000025f);
//                optimizerConv.setLearningRate(0.000025f);
//            }

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}