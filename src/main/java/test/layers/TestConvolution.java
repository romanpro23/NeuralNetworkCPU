package test.layers;

import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.*;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNTensor4D;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class TestConvolution {
    public static void main(String[] args) throws IOException {
        NNTensor[] input = new NNTensor[6];
        NNTensor[] output = new NNTensor[input.length];

//        Initializer initializer = new Initializer.RandomNormal();
//
//        FileWriter writerOut = new FileWriter("output.txt");
//        FileWriter writerIn = new FileWriter("input.txt");
//        FileWriter writerWeight = new FileWriter("weight.txt");
//
//        for (int i = 0; i < input.length; i++) {
//            input[i] = new NNTensor(32, 32, 64);
//            initializer.initialize(input[i]);
//            output[i] = new NNTensor(32, 32, 64);
//            initializer.initialize(output[i]);
//
//            input[i].save(writerIn);
//            output[i].save(writerOut);
//        }

        Scanner readerOut = new Scanner(new File("output.txt"));
        Scanner readerIn = new Scanner(new File("input.txt"));
        Scanner readerWeight = new Scanner(new File("weight.txt"));

        for (int i = 0; i < input.length; i++) {
            input[i] = NNTensor.read(readerIn);
            output[i] = NNTensor.read(readerOut);
        }

        SelfAttentionLayer layer = new SelfAttentionLayer(8);
        layer.initialize(new int[]{32, 32, 64});

        //layer.getWeight().save(writerWeight);
//        layer.setWeight(NNTensor4D.read(readerWeight));
        Optimizer optimizer = new AdamOptimizer();
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
