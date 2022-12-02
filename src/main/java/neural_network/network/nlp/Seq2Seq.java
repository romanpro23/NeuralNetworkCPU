package neural_network.network.nlp;

import neural_network.layers.recurrent.RecurrentNeuralLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.network.GAN.GAN;
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
        decoderLayer.setPreLayer(encoderLayer);
        encoder.queryTrain(input);

        float accuracy = decoder.train(output, NNArrays.toHotVectorNLP(output, sizeVocabulary));
        encoder.train(NNArrays.empty(NNArrays.isMatrix(encoder.getOutputs())));

        return accuracy;
    }

    public NNVector query(NNVector input) {
        decoderLayer.setPreLayer(encoderLayer);
        encoder.query(new NNVector[]{input});

        NNVector output = new NNVector(maxLength);
        NNVector start = new NNVector(1);

        decoder.query(new NNVector[]{start});

        int index = 0;
        decoderLayer.setReturnOwnState(true);

        int word = decoder.getOutputs()[0].indexMaxElement();
        while (index < (maxLength - 1) && word != 1) {
            output.set(index, word);
            decoder.query(new NNVector[]{new NNVector(new float[]{(float)word})});

            word = decoder.getOutputs()[0].indexMaxElement();
            index++;
        }
        output.set(index, 1);
        decoderLayer.setReturnOwnState(false);

        return output;
    }
}
