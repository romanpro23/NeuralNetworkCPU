package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.ciraf.Ciraf10Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.*;
import neural_network.regularization.Regularization;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestVGG10 {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new AdaBeliefOptimizer(0.0001);
//        NeuralNetwork vgg = new VGG()
//                .addInputLayer(32, 32, 3)
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
//                .addDenseLayer(512, new FunctionActivation.ReLU())
//                .addDropoutLayer(0.25)
//                .addDenseLayer(512, new FunctionActivation.ReLU())
//                .addDropoutLayer(0.25)
//                .addDenseLayer(100, new FunctionActivation.Softmax())
//                .createVGG()
//                .setFunctionLoss(new FunctionLoss.MSE())
//                .setOptimizer(optimizer)
//                .create();

        NeuralNetwork vgg = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/ciraf/vgg13.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();

        optimizer.read(new Scanner(new File("D:/NetworkTest/ciraf/vgg13_optimizer.txt")));

//        for (int i = 0; i < vgg.getLayers().size(); i++) {
//            if(vgg.getLayers().get(i) instanceof ActivationLayer3D){
//                ((ActivationLayer3D) vgg.getLayers().get(i)).setFunctionActivation(new FunctionActivation.ReLU());
//            }
//        }

        vgg.info();

        Ciraf100Loader3D loader = new Ciraf100Loader3D(new TransformData.Tanh()).useReverse();

        DataTrainer trainer = new DataTrainer(2500, 500, loader);

        for (int i = 0; i < 1000; i++) {
            long start = System.nanoTime();

            vgg.save(new FileWriter("D:/NetworkTest/ciraf/vgg13.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/ciraf/vgg13_optimizer.txt"));
            trainer.train(vgg, 60, 1, new DataMetric.Top1());
            trainer.score(vgg, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
