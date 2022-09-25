package neural_network.network.autoencoders;

import neural_network.layers.dense.VariationalLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;

public class VariationalAutoencoder {
    private final NeuralNetwork decoder;
    private final NeuralNetwork encoder;

    private final VariationalLayer variationalLayer;

    public VariationalAutoencoder(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.decoder = decoder;
        variationalLayer = ((VariationalLayer) encoder.getLayers().get(encoder.getLayers().size() - 1));
        this.encoder = encoder;
    }

    public NNArray[] query(NNArray[] input) {
        return decoder.query(encoder.query(input));
    }

    public NNArray[] queryDecoder(NNArray[] input) {
        return decoder.query(input);
    }

    public NNArray[] queryVariational(NNArray[] input) {
        variationalLayer.setRandomVariational(false);
        NNArray[] output = decoder.query(encoder.query(input));
        variationalLayer.setRandomVariational(true);
        return output;
    }

    public float train(NNArray[] input) {
        return train(input, input);
    }

    public float train(NNArray[] input, NNArray[] output) {
        encoder.queryTrain(input);
        decoder.train(encoder.getOutputs(), output);
        encoder.train(decoder.getError());

        return decoder.accuracy(output) + variationalLayer.findKLDivergence();
    }
}
