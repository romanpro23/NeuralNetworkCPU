package test.nlp;

import data.network_train.NNData1D;
import data.nlp.EnUaTranslateLoader;
import neural_network.layers.convolution_2d.SoftmaxLayer2D;
import neural_network.layers.recurrent.*;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.Seq2Seq;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;

import java.io.FileWriter;
import java.io.IOException;

public class TestSeq2Seq {
    public static void main(String[] args) throws IOException {
        EnUaTranslateLoader loader = new EnUaTranslateLoader(10000, 10000);
        Optimizer optEncoder, optDecoder;
        LSTMLayer gru;
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(10000, 64).setTrainable(true))
                .addLayer(new LSTMLayer(128, 0.2, true).setTrainable(true))
                .addLayer(gru = new LSTMLayer(128, 0.2, true).setTrainable(true))
                .setOptimizer(optEncoder = new AdamOptimizer())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new EmbeddingLayer(10000, 64).setTrainable(true))
                .addLayer(new LSTMBahdAttentionLayer(128, 128,0.2, true)
                        .setPreLayer(gru))
                .addLayer(new DenseTimeLayer(10000).setTrainable(true))
                .addLayer(new SoftmaxLayer2D())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optDecoder = new AdamOptimizer())
                .create();

        encoder.info();
        decoder.info();

        Seq2Seq model = new Seq2Seq(encoder, decoder);

        for (int i = 0; i < 10000; i++) {
            NNData1D train = loader.getNextTrainData(4);
            System.out.println(i + " - " + model.train(train.getInput(), train.getOutput()));

            if (i % 25 == 0) {
                encoder.save(new FileWriter("D:/NetworkTest/NLP/_encoder_lstm.txt"));
                optEncoder.save(new FileWriter("D:/NetworkTest/NLP/_encoder_lstm_optimizer.txt"));

                decoder.save(new FileWriter("D:/NetworkTest/NLP/_decoder_lstm.txt"));
                optDecoder.save(new FileWriter("D:/NetworkTest/NLP/_decoder_lstm_optimizer.txt"));
            }
        }
    }
}
