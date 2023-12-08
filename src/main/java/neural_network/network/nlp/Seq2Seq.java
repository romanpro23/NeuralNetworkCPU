package neural_network.network.nlp;

import neural_network.layers.recurrent.RecurrentNeuralLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

public class Seq2Seq {
    protected NeuralNetwork encoder;
    protected NeuralNetwork decoder;

    protected RecurrentNeuralLayer encoderLayer;
    protected RecurrentNeuralLayer decoderLayer;

    protected final int sizeVocabulary;
    protected int maxLength;

    public Seq2Seq(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.decoder = decoder;
        this.encoder = encoder;

        encoderLayer = (RecurrentNeuralLayer) encoder.getLayers().get(encoder.getLayers().size() - 1);
        decoderLayer = (RecurrentNeuralLayer) decoder.getLayers().get(1);

        this.sizeVocabulary = ((EmbeddingLayer) decoder.getLayers().get(0)).getSizeVocabulary();
        this.maxLength = 25;
    }

    public Seq2Seq setMaxLength(int size) {
        this.maxLength = size;

        return this;
    }

    public float train(NNArray[] input, NNArray[] output) {
        encoder.queryTrain(input);

        float accuracy = 0;
        //float accuracy = decoder.train(getInputDecoder(NNArrays.isVector(output)), getOutputDecoder(NNArrays.isVector(output)));
        encoder.train(NNArrays.empty(NNArrays.isMatrix(encoder.getOutputs())));

        return accuracy;
    }

    private NNVector[] getInputDecoder(NNVector[] input) {
        NNVector[] output = new NNVector[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i].subVector(0, input[i].size() - 1);
        }

        return output;
    }

    private NNMatrix[] getOutputDecoder(NNVector[] input) {
        NNVector[] output = new NNVector[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i].subVector(1, input[i].size() - 1);
        }

        return NNArrays.toHotVector(output, sizeVocabulary);
    }

    public NNVector query(NNVector input) {
        encoder.query(new NNVector[]{input});

        NNVector output = new NNVector(maxLength);
        NNVector start = new NNVector(1);

        decoder.query(new NNVector[]{start});

        int index = 0;

        int word = decoder.getOutputs()[0].indexMaxElement();
        while (index < (maxLength - 1) && word != 1) {
            output.set(index, word);
            decoder.query(new NNVector[]{new NNVector(new float[]{(float) word}, new short[]{(short) word})});

            word = decoder.getOutputs()[0].indexMaxElement();
            index++;
        }
        output.set(index, 1);

        return output;
    }
}
