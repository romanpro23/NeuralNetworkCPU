package test.layers;

import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

public class TestMaxPool {
    public static void main(String[] args) {
        NNTensor[] input = new NNTensor[64];
        NNTensor[] output = new NNTensor[input.length];

        Initializer initializer = new Initializer.RandomNormal();

        for (int i = 0; i < input.length; i++) {
            input[i] = new NNTensor(32, 32, 64);
            initializer.initialize(input[i]);
            output[i] = new NNTensor(32, 32, 64);

            initializer.initialize(output[i]);
        }

//        MaxPoolingLayer layer = new MaxPoolingLayer(2);
//        NNTensor[] in = new NNTensor[]{new NNTensor(4, 4, 1, new float[]{
//                1, 3, 5, 7,
//                2, 6, 1, 3,
//                5, 2, 0, 9,
//                4, 1, 3, 7})
//        };
        ConvolutionTransposeLayer layer = new ConvolutionTransposeLayer(64, 3, 1, 1);
        layer.initialize(new int[]{32, 32, 64});
        Optimizer optimizer = new AdaBeliefOptimizer(0.01);
        layer.initialize(optimizer);

        FunctionLoss loss = new FunctionLoss.MSE();
//
        for (int i = 0; i < 128; i++) {
            long start = System.nanoTime();
            layer.generateOutput(input);
            System.out.println(loss.findAccuracy(layer.getOutput(), output));
            layer.generateError(NNArrays.toTensor(loss.findDerivative(layer.getOutput(), output), 32, 32, 64));
            optimizer.update();
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
