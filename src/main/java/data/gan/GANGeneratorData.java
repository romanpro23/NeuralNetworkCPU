package data.gan;

import data.network_train.NNData;
import data.network_train.NNData1D;
import data.network_train.NNData2D;
import data.network_train.NNData3D;
import lombok.SneakyThrows;
import nnarrays.*;

public class GANGeneratorData {
    @SneakyThrows
    public static NNData generateData(NNArray[] real, NNArray[] fake) {
        NNVector[] output = new NNVector[real.length + fake.length];

        if (real[0].getCountAxes() == 1) {
            NNVector[] input = new NNVector[output.length];
            NNVector[] realV = NNArrays.isVector(real);
            NNVector[] fakeV = NNArrays.isVector(fake);
            for (int i = 0; i < real.length; i++) {
                input[i * 2] = realV[i];
                output[i * 2] = new NNVector(new float[]{1});
                input[i * 2 + 1] = fakeV[i];
                output[i * 2 + 1] = new NNVector(new float[]{0});
            }

            return new NNData1D(input, output);
        } else if (real[0].getCountAxes() == 2) {
            NNMatrix[] input = new NNMatrix[output.length];
            NNMatrix[] realV = NNArrays.isMatrix(real);
            NNMatrix[] fakeV = NNArrays.isMatrix(fake);
            for (int i = 0; i < real.length; i++) {
                input[i * 2] = realV[i];
                output[i * 2] = new NNVector(new float[]{1});
                input[i * 2 + 1] = fakeV[i];
                output[i * 2 + 1] = new NNVector(new float[]{0});
            }

            return new NNData2D(input, output);
        } else if (real[0].getCountAxes() == 3) {
            NNTensor[] input = new NNTensor[output.length];
            NNTensor[] realV = NNArrays.isTensor(real);
            NNTensor[] fakeV = NNArrays.isTensor(fake);
            for (int i = 0; i < real.length; i++) {
                input[i * 2] = realV[i];
                output[i * 2] = new NNVector(new float[]{1});
                input[i * 2 + 1] = fakeV[i];
                output[i * 2 + 1] = new NNVector(new float[]{0});
            }

            return new NNData3D(input, output);
        }

        throw new Exception("Error dimension input data!");
    }
}
