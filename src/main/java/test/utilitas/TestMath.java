package test.utilitas;

import neural_network.initialization.Initializer;
import nnarrays.NNMatrix;

public class TestMath {
    public static void main(String[] args) {
        NNMatrix a = new NNMatrix(512, 1024);
        NNMatrix b = new NNMatrix(1024, 1024);
        new Initializer.RandomUniform().initialize(a);
        new Initializer.RandomUniform().initialize(b);

        for (int i = 0; i < 100; i++) {
            long start = System.nanoTime();
            a.dot(b.transpose()).transpose();
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
