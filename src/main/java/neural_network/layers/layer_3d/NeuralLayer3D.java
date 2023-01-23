package neural_network.layers.layer_3d;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

public abstract class NeuralLayer3D extends NeuralLayer {
    protected int height, outHeight;
    protected int width, outWidth;
    protected int depth, outDepth;

    protected NNTensor[] input;
    protected NNTensor[] output;
    protected NNTensor[] error;
    protected NNTensor[] errorNL;

    @Override
    public int[] size() {
        return new int[]{outHeight, outWidth, outDepth};
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
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize element
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    public NNTensor[] getErrorNextLayer(NNArray[] error) {
        NNTensor[] errorNL = NNArrays.isTensor(error);

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
