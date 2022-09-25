package neural_network.layers.dense;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.util.ArrayList;
import java.util.Scanner;

public abstract class DenseNeuralLayer extends NeuralLayer {
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
    public void update(Optimizer optimizer) {
        //no have update element
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    public NNVector[] getErrorNextLayer(NNArray[] error){
        if(error != null){
            return NNArrays.isVector(error);
        }
        if(nextLayers.size() > 1) {
            NNVector[] errorNL = new NNVector[output.length];

            for (int i = 0; i < errorNL.length; i++) {
                errorNL[i] = new NNVector(countNeuron);
                for (NeuralLayer nextLayer : nextLayers) {
                    for (int k = 0; k < errorNL.length; k++) {
                        errorNL[i].getData()[k] += nextLayer.getError()[i].getData()[k];
                    }
                }
            }

            return errorNL;
        } else {
            return NNArrays.isVector(nextLayers.get(0).getError());
        }
    }
}
