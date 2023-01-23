package test.layers;

import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.*;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.*;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.IOException;

public class TestConvolution {
    public static void main(String[] args) throws IOException {
        NNTensor[] input = new NNTensor[12];
        NNTensor[] output = new NNTensor[input.length];

        Initializer initializer = new Initializer.RandomNormal();

//        FileWriter writerOut = new FileWriter("output.txt");
//        FileWriter writerIn = new FileWriter("input.txt");
//        FileWriter writerWeight = new FileWriter("weightAttention.txt");

        int size = 16;
        int depth = 128;

        for (int i = 0; i < input.length; i++) {
            input[i] = new NNTensor(size, size, depth);
            initializer.initialize(input[i]);
//            input[i].relu(input[i]);
            output[i] = new NNTensor(size, size, depth);
            initializer.initialize(output[i]);

//            input[i].save(writerIn);
//            output[i].save(writerOut);
        }

//        Scanner readerOut = new Scanner(new File("output.txt"));
//        Scanner readerIn = new Scanner(new File("input.txt"));
//        Scanner readerWeight = new Scanner(new File("weightAttention.txt"));
//
//        for (int i = 0; i < input.length; i++) {
//            input[i] = NNTensor.read(readerIn);
//            output[i] = NNTensor.read(readerOut);
//        }

        ConvolutionLayer layer = new ConvolutionLayer(depth,5,1,2).setTrainable(true);
        layer.initialize(new int[]{size, size, depth});

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
            layer.generateError(NNArrays.toTensor(loss.findDerivative(layer.getOutput(), output), size, size, depth));
            optimizer.update();
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
