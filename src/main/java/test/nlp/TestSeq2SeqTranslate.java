package test.nlp;

import data.network_train.NNData1D;
import data.nlp.EnUaTranslateLoader;
import neural_network.layers.convolution_2d.SoftmaxLayer2D;
import neural_network.layers.recurrent.DenseTimeLayer;
import neural_network.layers.recurrent.LSTMLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.Seq2Seq;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class TestSeq2SeqTranslate {
    public static void main(String[] args) throws Exception {
        EnUaTranslateLoader loader = new EnUaTranslateLoader(10000, 10000);
        NeuralNetwork encoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/_encoder_lstm.txt")))
                .create();

        NeuralNetwork decoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/_decoder_lstm.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

        encoder.info();
        decoder.info();

        Seq2Seq model = new Seq2Seq(encoder, decoder);
        System.out.println("Enter string: ");
        Scanner scanner = new Scanner(System.in);
        String str = scanner.nextLine();

        NNVector output = model.query(loader.getUaVector(str.split(" ")));
        System.out.println(loader.decodeUaString(output));
    }
}
