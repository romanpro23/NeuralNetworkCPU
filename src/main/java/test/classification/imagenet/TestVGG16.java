package test.classification.imagenet;

import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.MomentumOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestVGG16 {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new MomentumOptimizer(0.01, 0.9);
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
//                .setStopGradient(30)
                .setOptimizer(optimizer)
//                .addOptimizer(optimizerConv, 0, 30)
                .create();

//        NeuralNetwork vgg16 = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16_conv.txt")))
//                .setTrainable(true)
//                .setStopGradient(30)
//                .addOptimizer(optimizer, 30)
////                .addOptimizer(optimizer, 0, 30)
//                .setFunctionLoss(new FunctionLoss.CategoricalCrossEntropy())
//                .create();

//        for (int i = 0; i < vgg16.getLayers().size(); i++) {
//            if(vgg16.getLayers().get(i) instanceof DenseLayer){
//                ((DenseLayer) vgg16.getLayers().get(i)).setTrainable(false);
//            }
//        }

        vgg16.info();
//        optimizer.read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16_dense_optimizer.txt")));
//        optimizer.read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16_conv_optimizer.txt")));

        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useReverse().useCrop();
        DataTrainer trainer = new DataTrainer(5120/4, 512/4, loader);

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