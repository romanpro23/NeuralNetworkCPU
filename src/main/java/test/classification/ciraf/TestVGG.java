package test.classification.ciraf;

import data.ciraf.Ciraf100Loader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.VGG;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.Optimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;

public class TestVGG {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork vgg10 = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg10.txt")))
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(new AdamOptimizer())
//                .create();

//        Optimizer optimizerConv = new AdaBeliefOptimizer(0.0001);
        Optimizer optimizer = new AdaBeliefOptimizer(0.001);
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
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(512, new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addDenseLayer(100, new FunctionActivation.Softmax())
                .createVGG()
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();

        vgg16.info();

        Ciraf100Loader3D loader = new Ciraf100Loader3D(new TransformData.Tanh()).useReverse();

        DataTrainer trainer = new DataTrainer(5120/2, 512/2, loader);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();

            trainer.train(vgg16, 64, 1, 2, new DataMetric.Top1());
            trainer.score(vgg16, new DataMetric.Top1());
            vgg16.save(new FileWriter("D:/NetworkTest/ciraf/vgg13.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/ciraf/vgg13_optimizer.txt"));
//            optimizerConv.save(new FileWriter("D:/NetworkTest/ciraf/vgg13_conv_optimizer.txt"));

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
