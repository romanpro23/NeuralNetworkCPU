package neural_network.layers.doubles;

import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_2d.NeuralLayer2D;
import nnarrays.NNArray;

public abstract class DoubleNeuralLayer2D extends NeuralLayer2D {
    public abstract void generateOutput(NNArray[] inputFirst, NNArray[] inputSecond);

    public abstract NNArray[] getErrorFirst();

    public abstract NNArray[] getErrorSecond();
}