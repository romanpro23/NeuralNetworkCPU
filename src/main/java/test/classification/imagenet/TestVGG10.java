package test.classification.imagenet;

import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestVGG10 {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg16 = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg8.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdamOptimizer())
                ;
        vgg16.getLayers().add(2, new ConvolutionLayer(32, 3, 1, 1).setInitializer(new Initializer.HeNormal()));
        vgg16.getLayers().add(3, new ActivationLayer3D(new FunctionActivation.LeakyReLU()));

        vgg16.create();

        TinyImageNetLoader3D loader = new TinyImageNetLoader3D(new TransformData.Tanh());
        vgg16.info();

        DataTrainer trainer = new DataTrainer(1500, 300, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
//            vgg16.save(new FileWriter("D:/NetworkTest/Imagenet/vgg8.txt"));
//            trainer.train(vgg16, 60, 1, new DataMetric.Top1());
            trainer.score(vgg16, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
