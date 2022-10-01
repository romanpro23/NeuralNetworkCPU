package test.layers;

import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import neural_network.optimizers.SGDOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.util.Arrays;

public class TestBatchNorm {
    public static void main(String[] args) {
        NNVector[] input = new NNVector[4];
        NNVector[] error = new NNVector[input.length];

        input[0] = new NNVector(new float[]{4, 2, 1, 5});
        input[1] = new NNVector(new float[]{3, 5, -1, 7});
        input[2] = new NNVector(new float[]{-2, 3, 8, -5});
        input[3] = new NNVector(new float[]{9, -3, 0, 6});

        error[0] = new NNVector(new float[]{1, 2, 2, -1});
        error[1] = new NNVector(new float[]{3, -1, -2, 3});
        error[2] = new NNVector(new float[]{1, 4, -1, 2});
        error[3] = new NNVector(new float[]{2, 2, 5, -3});


        Optimizer optimizer = new AdamOptimizer();

        BatchNormalizationLayer layer = new BatchNormalizationLayer();
        layer.initialize(new int[]{4});
        layer.initialize(optimizer);


        for (int i = 0; i < 1; i++) {
            long start = System.nanoTime();
            optimizer.update();
            layer.generateTrainOutput(input);
            System.out.println("Outputs");
            for (int j = 0; j < layer.getOutput().length; j++) {
                System.out.println(Arrays.toString(layer.getOutput()[j].getData()));
            }
            layer.generateError(error);
            optimizer.update();
            System.out.println("Errors");
            for (int j = 0; j < layer.getOutput().length; j++) {
                System.out.println(Arrays.toString(layer.getError()[j].getData()));
            }
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
