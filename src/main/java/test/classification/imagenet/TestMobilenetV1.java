package test.classification.imagenet;

import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.classification.MobileNet;
import neural_network.optimizers.AdaBeliefOptimizer;
import trainer.DataMetric;
import trainer.DataTrainer;

import java.io.FileWriter;
import java.io.IOException;

public class TestMobilenetV1 {
    public static void main(String[] args) throws IOException {
        AdaBeliefOptimizer optimizer = new AdaBeliefOptimizer(0.0001);
        double scaleWidth = 1;
        NeuralNetwork mobilenetV1 = new MobileNet()
                .addInputLayer(64, 64, 3)
                .addConvolutionLayer((int) (32 * scaleWidth), 3, 2)
                .addDWConvolutionLayer((int) (32 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (64 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (64 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (128 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (128 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (128 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (128 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addGlobalAveragePoolingLayer()
                .addDropoutLayer(0.2)
                .addDenseLayer(200, new FunctionActivation.Softmax())
                .createMobilenet()
//                .createMobileNetV1(64,64, 3, 0.5, 200,0.2)
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();

        TinyImageNetLoader3D loader = new TinyImageNetLoader3D(new TransformData.Tanh().addNoise()).useReverse().useCrop();
        mobilenetV1.info();

        DataTrainer trainer = new DataTrainer(1000, 100, loader);

        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            mobilenetV1.save(new FileWriter("D:/NetworkTest/Imagenet/mobilenetV1_0.5.txt"));
            optimizer.save(new FileWriter("D:/NetworkTest/Imagenet/mobilenetV1_0.5_optimizer.txt"));
            trainer.train(mobilenetV1, 60, 1, new DataMetric.Top1());
            trainer.score(mobilenetV1, new DataMetric.Top1());

            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
