package neural_network.layers.layer_2d;

import lombok.SneakyThrows;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

public abstract class NeuralLayer2D extends NeuralLayer {
    protected int width, outWidth;
    protected int depth, outDepth;

    protected NNMatrix[] input;
    protected NNMatrix[] output;
    protected NNMatrix[] error;
    protected NNMatrix[] errorNL;

    @Override
    public int[] size() {
        return new int[]{outWidth, outDepth};
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        width = size[0];
        depth = size[1];
        outWidth = width;
        outDepth = depth;
    }

    @Override
    public void initialize(Optimizer optimizer) {

    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    public NNMatrix[] getErrorNextLayer(NNArray[] error) {
        NNMatrix[] errorNL = NNArrays.isMatrix(error);

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
