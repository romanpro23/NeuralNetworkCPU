package neural_network.network.autoencoders;


import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNVector;

public class Autoencoder {
    private final NeuralNetwork decoder;
    private final NeuralNetwork encoder;

    public Autoencoder(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.decoder = decoder;
        this.encoder = encoder;
    }

    public NNArray[] query(NNArray[] input){
        return decoder.query(encoder.query(input));
    }

    public float train(NNArray[]input){
        return train(input, input);
    }

    public float train(NNArray[] input, NNArray[] output){
        encoder.queryTrain(input);
        decoder.train(encoder.getOutputs(), output);
        encoder.train(decoder.getError());

        return decoder.accuracy(output);
    }
}
