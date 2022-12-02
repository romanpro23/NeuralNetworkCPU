package neural_network.layers.dual;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;

public abstract class DualNeuralLayer extends NeuralLayer {
    @Override
    public void generateOutput(NNArray[] input) {
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize elements
    }

    public abstract void generateOutput(NNArray[] input1, NNArray[] input2);

    public abstract NNArray[] getErrorFirst();

    public abstract NNArray[] getErrorSecond();
}
