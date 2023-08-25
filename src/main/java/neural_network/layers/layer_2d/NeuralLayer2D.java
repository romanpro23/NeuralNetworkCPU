package neural_network.layers.layer_2d;

import lombok.SneakyThrows;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import utilities.CublasUtil;

public abstract class NeuralLayer2D extends NeuralLayer {
    protected int width, outWidth;
    protected int depth, outDepth;

    protected NNMatrix[] input;
    protected NNMatrix[] output;
    protected NNMatrix[] error;
    protected NNMatrix[] errorNL;

    protected CublasUtil.Matrix[] input_gpu;
    protected CublasUtil.Matrix[] output_gpu;
    protected CublasUtil.Matrix[] error_gpu;
    protected CublasUtil.Matrix[] errorNL_gpu;

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
        //no have initialize element
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

    public CublasUtil.Matrix[] getErrorNextLayer(CublasUtil.Matrix[] error) {
        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < error.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    error[i].add(nextLayer.getErrorNL_gpu()[i]);
                }
            }
        }
        return error;
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return output_gpu;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return error_gpu;
    }

    public abstract void generateError(CublasUtil.Matrix[] errors);
}
