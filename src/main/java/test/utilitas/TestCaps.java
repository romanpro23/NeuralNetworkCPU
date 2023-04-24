package test.utilitas;

import nnarrays.NNMatrix;
import nnarrays.NNTensor;

public class TestCaps {
    public static void main(String[] args) {
        NNTensor input = new NNTensor(2, 6, 4);
        input.fill(1);
        NNMatrix c = new NNMatrix(2, 4);
        c.fill(4);
        c.set(5, -2);
        NNMatrix b = new NNMatrix(2, 4);
        b.addScalarMul(input, c);
        System.out.println(b.toString());
    }
}
