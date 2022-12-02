package neural_network.layers.dense;

import lombok.Getter;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.util.ArrayList;
import java.util.Scanner;

public abstract class DenseNeuralLayer extends NeuralLayer {
    @Getter
    protected int countNeuron;
    protected NNVector[] input;
    protected NNVector[] output;
    protected NNVector[] error;
    protected NNVector[] errorNL;

    @Override
    public int[] size() {
        return new int[]{countNeuron};
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize element
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    public NNVector[] getErrorNextLayer(NNArray[] error) {
        NNVector[] errorNL = NNArrays.isVector(error);

        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < errorNL.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    errorNL[i].add(nextLayer.getErrorNL()[i]);
                }
            }
        }
        return errorNL;
    }
}
