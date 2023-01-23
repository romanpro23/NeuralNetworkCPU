package test.utilitas;

import neural_network.initialization.Initializer;
import nnarrays.NNTensor;
import nnarrays.NNTensor4D;

public class TestConvolution {
    public static void main(String[] args) {
        NNTensor input = new NNTensor(16, 16, 256);
        new Initializer.RandomNormal().initialize(input);
        NNTensor output = new NNTensor(16, 16, 256);
        new Initializer.RandomNormal().initialize(output);
        NNTensor4D weight = new NNTensor4D(256, 3, 3, 256);
        new Initializer.RandomNormal().initialize(weight);

        int iteration = 100;

        long sum = 0;

        for (int i = 0; i < iteration; i++) {
            long start = System.nanoTime();
            weight.convolution(input, output, 1, 1, 1);
            sum += (System.nanoTime() - start);
            System.out.println((System.nanoTime() - start) / 1000000);
        }
        System.out.println((sum / iteration) / 1000);
    }
}
