package neural_network.layers.layer_1d;

import lombok.Getter;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

public abstract class DenseNeuralLayer extends NeuralLayer {
    @Getter
    protected int countNeuron;
    protected int countNeuronHide;
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
