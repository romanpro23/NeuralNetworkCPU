package test.utilitas;

import neural_network.initialization.Initializer;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

public class TestMath {
    public static void main(String[] args) {
        NNVector vector = new NNVector(16);
        new Initializer.RandomUniform().initialize(vector);
        System.out.println(vector.mod());
        NNVector v1 = vector.squash();
        System.out.println(v1.mod());
        NNVector v2 = v1.squash();
        System.out.println(v2.mod());

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
