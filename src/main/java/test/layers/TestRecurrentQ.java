package test.layers;

import neural_network.initialization.Initializer;
import neural_network.layers.recurrent.LSTMLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
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
            input[i].set(0, 1, 0.6f);
            //initializerInput.initialize(input[i]);
            output[i] = new NNMatrix(4, 4);
            output[i].fill(1f);

//            initializerInput.initialize(output[i]);
        }

        output[0].softmax(input[0]);
        System.out.println(Arrays.toString(output[0].getData()));

        LSTMLayer layer = new LSTMLayer(4, 0, true);
        layer.initialize(new int[]{4, 4});
        Optimizer optimizer = new AdamOptimizer();
        layer.initialize(optimizer);

        FunctionLoss loss = new FunctionLoss.MSE();
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
