package test.nlp;

import data.network_train.NNData1D;
import data.nlp.UaFictionLoader;
import neural_network.layers.layer_2d.DenseLayer2D;
import neural_network.layers.layer_2d.SoftmaxLayer2D;
import neural_network.layers.recurrent.*;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.Seq2Seq;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;

import java.io.FileWriter;

public class TestSeq2SeqUa {
    public static void main(String[] args) throws Exception {
        UaFictionLoader loader = new UaFictionLoader(1000);

        Optimizer optEncoder, optDecoder;
        Bidirectional gru;
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(1000, 64).setTrainable(true))
                .addLayer(gru = new Bidirectional(new RecurrentLayer(128, 0.2, true).setTrainable(true)))
                .setOptimizer(optEncoder = new AdamOptimizer())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(1000, 64).setTrainable(true))
                .addLayer(new RecurrentLayer(256, 0.2, true))
                .addLayer(new DenseLayer2D(1000).setTrainable(true))
                .addLayer(new SoftmaxLayer2D())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optDecoder = new AdamOptimizer())
                .create();

        encoder.info();
        decoder.info();

        Seq2Seq model = new Seq2Seq(encoder, decoder);

        for (int i = 0; i < 10000; i++) {
            NNData1D train = loader.getNextTrainData(64);
            System.out.println(i + " - " + model.train(train.getInput(), train.getOutput()));

            if (i % 25 == 0) {
                encoder.save(new FileWriter("D:/NetworkTest/NLP/_encoder_rnn.txt"));
                optEncoder.save(new FileWriter("D:/NetworkTest/NLP/_encoder_rnn_optimizer.txt"));

                decoder.save(new FileWriter("D:/NetworkTest/NLP/_decoder_rnn.txt"));
                optDecoder.save(new FileWriter("D:/NetworkTest/NLP/_decoder_rnn_optimizer.txt"));
            }
        }
    }
}
