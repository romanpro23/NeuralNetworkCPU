package test.nlp;

import data.nlp.PositionUaLoader;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.CharRNN;
import nnarrays.NNVector;

import java.io.File;
import java.util.Scanner;

public class TestRealCharRNN {
    public static void main(String[] args) throws Exception {
        PositionUaLoader loader = new PositionUaLoader(50);
        NeuralNetwork charRNN = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/char-gru.txt"))).create();

        charRNN.info();

        CharRNN model = new CharRNN(charRNN);

        for (int i = 0; i < 100000; i++) {
            System.out.println("Enter string: ");
            String start = new Scanner(System.in).nextLine();
            System.out.println("Result: ");
            NNVector result = model.query(loader.codeString(start), 100);
            System.out.println(loader.decodeString(result));
        }
    }
}
