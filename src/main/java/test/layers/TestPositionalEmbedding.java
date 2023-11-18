package test.layers;

import data.ImageCreator;
import neural_network.layers.layer_2d.PositionalEmbeddingLayer;
import nnarrays.NNTensor;

public class TestPositionalEmbedding {
    public static void main(String[] args) {
        PositionalEmbeddingLayer layer = new PositionalEmbeddingLayer();
        layer.initialize(new int[]{100, 512});

        ImageCreator.drawImage(new NNTensor(100, 512, 1, layer.getPositionalVal().getData(), layer.getPositionalVal().getSdata()), 100, 512, "_position", "D:/NetworkTest/Transformer", true);
    }
}
