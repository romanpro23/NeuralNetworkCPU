package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.ciraf.Ciraf10Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class TestVGG {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork vgg10 = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg10.txt")))
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .create();

//        for (int i = vgg10.getLayers().size() - 1; i >= 0; i--) {
//            if (vgg10.getLayers().get(i) instanceof DropoutLayer3D) {
////                vgg10.getLayers().add(i, new ActivationLayer3D(new FunctionActivation.LeakyReLU()));
//                vgg10.getLayers().remove(i);
//            }
//        }
//        vgg10.setStopGradient(20)
//                .create();
//        vgg.setTrainable(false);
//
//        NeuralNetwork vgg10 = new NeuralNetwork()
//                .addInputLayer(32, 32, 3)
//                .addLayers(vgg.getConvolutionLayers())
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new BatchNormalizationLayer3D(0.9))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new Flatten3DLayer())
//                .addLayer(new DenseLayer(512).setInitializer(new Initializer.XavierNormal()))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addLayer(new DenseLayer(512).setInitializer(new Initializer.XavierNormal()))
//                .addActivationLayer(new FunctionActivation.ReLU())
//                .addDenseLayer(100, new FunctionActivation.Softmax())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .setStopGradient(23)
//                .create();

//        vgg10.create();
        NeuralNetwork vgg10 = new NeuralNetwork()
                .addInputLayer(64, 64, 3)
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new ConvolutionLayer(256, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new MaxPoolingLayer(2))
//                .addLayer(new ConvolutionLayer(512, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new ConvolutionLayer(512, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new ConvolutionLayer(512, 3, 1, 1).setInitializer(new Initializer.XavierNormal()))
//                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
//                .addLayer(new MaxPoolingLayer(2))
                .addLayer(new Flatten3DLayer())
                .addLayer(new DenseLayer(512).setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer(512).setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer(100).setInitializer(new Initializer.XavierNormal()))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdaBeliefOptimizer())
                .create();

        vgg10.info();

        Ciraf100Loader3D loader = new Ciraf100Loader3D(new TransformData.VGG());

        DataTrainer trainer = new DataTrainer(3000, 1000, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

//            vgg10.save(new FileWriter("D:/NetworkTest/ciraf/vgg10.txt"));
            trainer.train(vgg10, 60, 1, new DataMetric.Top1());
            trainer.score(vgg10, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
