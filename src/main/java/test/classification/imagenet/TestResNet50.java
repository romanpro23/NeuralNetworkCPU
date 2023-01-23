package test.classification.imagenet;

import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.ResNet;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;

public class TestResNet50 {
    public static void main(String[] args) throws Exception {
        Optimizer optimizer = new AdaBeliefOptimizer(0.0001);
        NeuralNetwork resnet50 = new ResNet()
                .addInputLayer(64, 64, 3)
                .addConvolutionLayer(32, 5, 2, 2)
                .addMaxPoolingLayer(3,2)
                .addBottleneckResBlock(128, true, 1)
                .addBottleneckResBlock(128)
                .addBottleneckResBlock(128)
                .addBottleneckResBlock(256, true)
                .addBottleneckResBlock(256)
                .addBottleneckResBlock(256)
                .addBottleneckResBlock(256)
                .addBottleneckResBlock(512, true)
                .addBottleneckResBlock(512)
                .addBottleneckResBlock(512)
                .addBottleneckResBlock(512)
                .addBottleneckResBlock(512)
                .addBottleneckResBlock(512)
                .addBottleneckResBlock(1024, true)
                .addBottleneckResBlock(1024)
                .addBottleneckResBlock(1024)
                .addGlobalAveragePoolingLayer()
                .addDenseLayer(250, new FunctionActivation.Softmax())
                .createResNet()
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();

//        NeuralNetwork resnet50 = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16.txt")))
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setTrainable(true)
//                .setOptimizer(optimizer)
//                .create();

//        optimizer.read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16_optimizer.txt")));

        TinyImageNetLoader3D loader = new TinyImageNetLoader3D(new TransformData.Tanh()).useCrop().useReverse();
        resnet50.info();

        DataTrainer trainer = new DataTrainer(1000, 100, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            resnet50.save(new FileWriter("D:/NetworkTest/Imagenet/resnet50.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/Imagenet/resnet50_optimizer.txt"));
            trainer.train(resnet50, 60, 1, new DataMetric.Top1());
            trainer.score(resnet50, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
