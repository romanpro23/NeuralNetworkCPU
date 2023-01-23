package test.nlp;

import data.nlp.PositionUaLoader;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.CharRNN;
import neural_network.optimizers.AdaBeliefOptimizer;
import neural_network.optimizers.Optimizer;

import java.io.*;
import java.util.Scanner;

public class TestCharRNN {
    public static void main(String[] args) throws Exception {
        PositionUaLoader loader = new PositionUaLoader(50);

        Optimizer optimizer = new AdaBeliefOptimizer();
        NeuralNetwork charRNN = NeuralNetwork.read(new Scanner(new File("D:/NetworkTest/NLP/char-gru.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();

//        NeuralNetwork charRNN = new NeuralNetwork()
//                .addInputLayer(100)
//                .addLayer(new EmbeddingLayer(50, 256))
//                .addLayer(new GRULayer(1024, 0, true))
//                .addLayer(new DenseLayer2D(50))
//                .addLayer(new SoftmaxLayer2D())
//                .setFunctionLoss(new FunctionLoss.CrossEntropy())
//                .setOptimizer(optimizer)
//                .setTrainable(true)
//                .create();

        optimizer.read(new Scanner(new File("D:/NetworkTest/NLP/char-gru_optimizer.txt")));

        charRNN.info();
        CharRNN model = new CharRNN(charRNN);

        for (int i = 0; i < 100000; i++) {
            System.out.println(i + " - " + model.train(loader.getNextTrainData(64).getInput()));
            if(i % 10 == 0){
                charRNN.save(new FileWriter("D:/NetworkTest/NLP/char-gru.txt"));
                optimizer.save(new FileWriter("D:/NetworkTest/NLP/char-gru_optimizer.txt"));
            }
        }
    }
}
