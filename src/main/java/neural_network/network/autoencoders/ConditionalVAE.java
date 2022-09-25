package neural_network.network.autoencoders;

import neural_network.layers.dense.VariationalLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;

public class ConditionalVAE {
    private final NeuralNetwork decoder;
    private final NeuralNetwork encoder;

    private final VariationalLayer variationalLayer;

    public ConditionalVAE(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.decoder = decoder;
        variationalLayer = ((VariationalLayer) encoder.getLayers().get(encoder.getLayers().size() - 1));
        this.encoder = encoder;
    }

    public NNArray[] query(NNArray[] input, NNArray[] label) {
        NNArray[] inputEncoder = NNArrays.concat(input, label);
        encoder.query(inputEncoder);
        NNArray[] inputDecoder = NNArrays.concat(encoder.getOutputs(), label);
        return decoder.query(inputDecoder);
    }

    public NNArray[] queryDecoder(NNArray[] input, NNArray[] label) {
        NNArray[] inputDecoder = NNArrays.concat(input, label);
        return decoder.query(inputDecoder);
    }

    public NNArray[] queryVariational(NNArray[] input, NNArray[] label) {
        variationalLayer.setRandomVariational(false);

        NNArray[] inputEncoder = NNArrays.concat(input, label);
        encoder.query(inputEncoder);

        NNArray[] inputDecoder = NNArrays.concat(encoder.getOutputs(), label);
        NNArray[] output = decoder.query(inputDecoder);

        variationalLayer.setRandomVariational(true);
        return output;
    }

    public float train(NNArray[] input, NNArray[] label) {
        return train(input, input, label);
    }

    public float train(NNArray[] input, NNArray[] output, NNArray[] label) {
        NNArray[] inputEncoder = NNArrays.concat(input, label);
        encoder.queryTrain(inputEncoder);

        NNArray[] inputDecoder = NNArrays.concat(encoder.getOutputs(), label);
        decoder.train(inputDecoder, output);

        NNArray[] errorEncoder = NNArrays.subArray(decoder.getError(), encoder.getOutputs());
        encoder.train(errorEncoder);

        return decoder.accuracy(output) + variationalLayer.findKLDivergence();
    }
}
