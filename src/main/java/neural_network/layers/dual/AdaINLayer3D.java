package neural_network.layers.dual;

import lombok.Getter;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;

public class AdaINLayer3D extends DualNeuralLayer {
    private int height, width, depth;
    private int outHeight, outWidth, outDepth;
    private int size;

    private NNTensor[] inputFirst, inputSecond;
    private NNTensor[] errorFirst, errorSecond;
    private NNTensor[] output;

    @Getter
    private NNVector[] meanFirst, meanSecond;
    @Getter
    private NNVector[] varFirst, varSecond;

    @Override
    public int[] size() {
        return new int[]{outHeight, outWidth, outDepth};
    }

    @Override
    public int info() {
        System.out.println("AdaIN\t\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("AdaIN layer 3D\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];
        outWidth = width;
        outHeight = height;
        outDepth = depth;

        this.size = height * width;
    }

    @Override
    public void generateOutput(NNArray[] input1, NNArray[] input2) {
        inputFirst = NNArrays.isTensor(input1);
        inputSecond = NNArrays.isTensor(input2);
        output = new NNTensor[input1.length];

        meanFirst = new NNVector[input1.length];
        meanSecond = new NNVector[input1.length];
        varFirst = new NNVector[input1.length];
        varSecond = new NNVector[input1.length];

        for (int i = 0; i < input1.length; i++) {
            meanFirst[i] = findMean(inputFirst[i]);
            meanSecond[i] = findMean(inputSecond[i]);

            varFirst[i] = findVariance(inputFirst[i], meanFirst[i]);
            varSecond[i] = findVariance(inputSecond[i], meanSecond[i]);

            output[i] = adaptiveNormalization(inputFirst[i], meanFirst[i], meanSecond[i], varFirst[i], varSecond[i]);
        }
    }

    private NNTensor adaptiveNormalization(NNTensor input, NNVector mean1, NNVector var1, NNVector mean2, NNVector var2) {
        NNTensor output = new NNTensor(input.shape());

        for (int i = 0, index = 0; i < size; i++) {
            for (int j = 0; j < depth; j++, index++) {
                output.getData()[index] = ((input.get(index) - mean1.get(j)) / var1.get(j)) * var2.get(j) + mean2.get(j);
            }
        }

        return output;
    }

    @Override
    public void generateError(NNArray[] error) {

    }

    public NNVector findMean(NNTensor input) {
        NNVector mean = new NNVector(depth);

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                mean.getData()[k] += input.getData()[index];
            }
        }
        mean.div(size);

        return mean;
    }

    public NNVector findVariance(NNTensor input, NNVector mean) {
        NNVector var = new NNVector(depth);
        float sub;
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                sub = input.getData()[index] - mean.getData()[k];
                var.getData()[k] += sub * sub;
            }
        }
        var.div(size);

        return var;
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return errorFirst;
    }

    @Override
    public NNArray[] getErrorFirst() {
        return errorFirst;
    }

    @Override
    public NNArray[] getErrorSecond() {
        return errorSecond;
    }
}
