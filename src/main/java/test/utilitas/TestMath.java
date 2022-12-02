package test.utilitas;

import neural_network.initialization.Initializer;
import nnarrays.NNMatrix;

public class TestMath {
    public static void main(String[] args) {
        NNMatrix a = new NNMatrix(4, 6);
        NNMatrix b = new NNMatrix(6, 6);
        new Initializer.RandomUniform().initialize(a);
        new Initializer.RandomUniform().initialize(b);

        for (int i = 0; i < 1; i++) {
            long start = System.nanoTime();
            System.out.println(a);
            System.out.println(b);
            System.out.println(a.dot(b));
            System.out.println(a.dot(b.transpose()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
