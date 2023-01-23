package neural_network.network.nlp;

import neural_network.layers.layer_2d.MultiHeadAttentionLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;

public class TransformerModel {
    private final NeuralNetwork encoder;
    private final NeuralNetwork decoder;

    protected final int sizeVocabulary;

    public TransformerModel(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.encoder = encoder;
        this.decoder = decoder;

        for (int i = 0; i < decoder.getLayers().size(); i++) {
            if (decoder.getLayer(i) instanceof MultiHeadAttentionLayer && !((MultiHeadAttentionLayer) decoder.getLayer(i)).isUseMask()) {
                ((MultiHeadAttentionLayer) decoder.getLayer(i)).addEncoderLayer(encoder.getLastLayer());
            }
        }

        this.sizeVocabulary = ((EmbeddingLayer) decoder.getLayers().get(0)).getSizeVocabulary();
    }

    public float train(NNArray[] input, NNArray[] output){
        encoder.queryTrain(input);
        float accuracy = decoder.train(input, NNArrays.toHotVector(NNArrays.isVector(output), sizeVocabulary));
        encoder.train(NNArrays.empty(NNArrays.isMatrix(encoder.getOutputs())));

        return accuracy;
    }
}
