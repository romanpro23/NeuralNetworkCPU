package test.nlp;

import data.nlp.UaFictionLoader;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.Seq2Seq;
import nnarrays.NNVector;

import java.io.File;
import java.util.Scanner;

public class TestSeq2SeqTranslate {
    public static void main(String[] args) throws Exception {
        UaFictionLoader loader = new UaFictionLoader(1000);
        NeuralNetwork encoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/_encoder_rnn.txt")))
                .create();

        NeuralNetwork decoder = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/_decoder_rnn.txt")))
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
