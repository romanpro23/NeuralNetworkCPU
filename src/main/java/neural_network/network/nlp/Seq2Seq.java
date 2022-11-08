package neural_network.network.nlp;

import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import nnarrays.NNVector;

public class Seq2Seq {
    protected NeuralNetwork encoder;
    protected NeuralNetwork decoder;

    public Seq2Seq(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.decoder = decoder;
        this.encoder = encoder;
    }

    public float train(NNVector[] input, NNVector[] output){
        return 0;
    }
}
