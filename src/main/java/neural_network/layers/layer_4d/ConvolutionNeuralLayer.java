package neural_network.layers.layer_4d;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor4D;

public abstract class ConvolutionNeuralLayer extends NeuralLayer {
    protected int length, outLength;
    protected int height, outHeight;
    protected int width, outWidth;
    protected int depth, outDepth;

    protected NNTensor4D[] input;
    protected NNTensor4D[] output;
    protected NNTensor4D[] error;
    protected NNTensor4D[] errorNL;

    @Override
    public int[] size() {
        return new int[]{outLength,outHeight, outWidth, outDepth};
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 4) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        length = size[0];
        height = size[1];
        width = size[2];
        depth = size[3];
        outWidth = width;
        outHeight = height;
        outDepth = depth;
        outLength = length;
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize element
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    public NNTensor4D[] getErrorNextLayer(NNArray[] error) {
        NNTensor4D[] errorNL = NNArrays.isTensor4D(error);

        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < errorNL.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    errorNL[i].add(nextLayer.getErrorNL()[i]);
                }
            }
        }
        return errorNL;
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }
}
