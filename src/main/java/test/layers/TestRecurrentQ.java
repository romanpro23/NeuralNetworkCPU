package test.layers;

import neural_network.initialization.Initializer;
import neural_network.layers.recurrent.RecurrentLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

import java.util.Arrays;

public class TestRecurrentQ {
    public static void main(String[] args) {
        NNMatrix[] input = new NNMatrix[1];
        NNMatrix[] output = new NNMatrix[input.length];

        Initializer initializer = new Initializer.RandomUniform();

        for (int i = 0; i < input.length; i++) {
            input[i] = new NNMatrix(4, 4);
            input[i].fill(0.1f);
            //initializer.initialize(input[i]);
            output[i] = new NNMatrix(4, 6);
            output[i].fill(1f);

//            initializer.initialize(output[i]);
        }

        RecurrentLayer layer = new RecurrentLayer(6, 0, true);
        layer.initialize(new int[]{4, 4});
        Optimizer optimizer = new AdamOptimizer();
        layer.initialize(optimizer);

        FunctionLoss loss = new FunctionLoss.Quadratic();
//
        for (int i = 0; i < 1; i++) {
            long start = System.nanoTime();
            layer.generateTrainOutput(input);
            System.out.println(Arrays.toString(layer.getOutput()[0].getData()));
            System.out.println(loss.findAccuracy(layer.getOutput(), output));
            layer.generateError(output);
            System.out.println(Arrays.toString(layer.getError()[0].getData()));
            optimizer.update();
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
