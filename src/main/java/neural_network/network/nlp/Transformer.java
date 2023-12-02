package neural_network.network.nlp;

import neural_network.activation.FunctionActivation;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.layer_2d.DenseLayer2D;
import neural_network.layers.layer_2d.MultiHeadAttentionLayer;
import neural_network.layers.layer_2d.PositionalEmbeddingLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.network.NeuralNetwork;

public class Transformer {
    private final NeuralNetwork transformer;
    private int depth;

    public Transformer() {
        transformer = new NeuralNetwork();
    }

    public Transformer addInputLayer(int width, int depth) {
        transformer.addInputLayer(width, depth);
        this.depth = depth;

        return this;
    }

    public Transformer addLayer(NeuralLayer layer) {
        transformer.addLayer(layer);

        return this;
    }

    public Transformer addEmbeddingLayer(int sizeVocabulary, int lengthVector) {
        transformer.addLayer(new EmbeddingLayer(sizeVocabulary, lengthVector));

        return this;
    }

    public Transformer addPositionalEmbedding() {
        transformer.addLayer(new PositionalEmbeddingLayer());

        return this;
    }

    public Transformer addEncoderBlock(int countHead) {
        return addEncoderBlock(countHead, depth, depth, (short)0, (short)0);
    }

    public Transformer addEncoderBlock(int countHead, int sizeDense) {
        return addEncoderBlock(countHead, depth, sizeDense, (short)0, (short)0);
    }

    public Transformer addEncoderBlock(int countHead, int sizeKey, int sizeDense) {
        return addEncoderBlock(countHead, sizeKey, sizeDense, (short)0, (short)0);
    }

    public Transformer addEncoderBlock(int countHead, int sizeKey, int sizeDense, short dropout) {
        return addEncoderBlock(countHead, sizeKey, sizeDense, (short)0, dropout);
    }

    public Transformer addEncoderBlock(int countHead, int sizeKey, int sizeDense, short dropoutAttention, short dropout) {
        transformer.addLayer(new AdditionBlock()
                .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention))
        ).addLayer(new NormalizationLayer2D());

        AdditionBlock additionBlock = new AdditionBlock()
                .addLayer(new DenseLayer2D(sizeDense, false))
                .addLayer(new ActivationLayer2D(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer2D(depth, false));

        if (dropout != 0) {
            additionBlock.addLayer(new DropoutLayer2D(dropout));
        }

        transformer.addLayer(additionBlock)
                .addLayer(new NormalizationLayer2D());

        return this;
    }

    public Transformer addDecoderBlock( int countHead) {
        return addDecoderBlock(countHead, depth, depth, (short)0, ((short)0));
    }

    public Transformer addDecoderBlock(int countHead, int sizeDense) {
        return addDecoderBlock(countHead, depth, sizeDense, (short)0, ((short)0));
    }

    public Transformer addDecoderBlock(int countHead, int sizeKey, int sizeDense) {
        return addDecoderBlock(countHead, sizeKey, sizeDense, (short)0, ((short)0));
    }

    public Transformer addDecoderBlock(int countHead, int sizeKey, int sizeDense, short dropout) {
        return addDecoderBlock(countHead, sizeKey, sizeDense, (short)0, dropout);
    }

    public Transformer addDecoderBlock(int countHead, int sizeKey, int sizeDense, short dropoutAttention, short dropout) {
        transformer.addLayer(new AdditionBlock()
                .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention).setMask())
        ).addLayer(new NormalizationLayer2D())
                .addLayer(new AdditionBlock()
                .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention))
        ).addLayer(new NormalizationLayer2D());

        AdditionBlock additionBlock = new AdditionBlock()
                .addLayer(new DenseLayer2D(sizeDense, false))
                .addLayer(new ActivationLayer2D(new FunctionActivation.ReLU()))
                .addLayer(new DenseLayer2D(depth, false));

        if (dropout != 0) {
            additionBlock.addLayer(new DropoutLayer2D(dropout));
        }

        transformer.addLayer(additionBlock)
                .addLayer(new NormalizationLayer2D());

        return this;
    }

    public Transformer addDenseLayer(int countNeuron){
        transformer.addLayer(new DenseLayer2D(countNeuron, false));

        return this;
    }

    public Transformer addSoftmax(){
        transformer.addLayer(new SoftmaxLayer2D());

        return this;
    }

    public NeuralNetwork createTransformer(){
        return transformer;
    }
}
