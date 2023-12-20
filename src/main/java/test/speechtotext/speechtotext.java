package test.speechtotext;

import data.network_train.NNData2D;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.layers.reshape.ReshapeLayer2D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import trainer.DataMetric;
import trainer.DataTrainer;
import utilities.GPUInit;

import java.io.File;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Scanner;

import static neural_network.layers.NeuralLayer.CallGarbageCollector;
import static nnarrays.NNArray.GetFirstSingleValueShortStatic;

public class speechtotext {
    static NeuralNetwork network;

    public static void main(String[] args) throws Exception {
        GPUInit.startup();

        PositionLoader loader = new PositionLoader(44);
        loader.setUseReverse(false);

        Optimizer optimizer = new AdamOptimizer();
        network = NeuralNetwork.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt")))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.MAE())
                .setTrainable(true)
                .create();

        /*network = new NeuralNetwork();
        network.addInputLayer(480, 24)
        .addLayer(new TYPE2Float2D())
        .addLayer(new VITPositionalEmbeddingLayer(false))
        .addLayer(new ConvolutionLayer(24, 24))
        .addLayer(new ConvolutionLayer(24, 24))
        .addLayer(new ConvolutionLayer(24, 24))
        .addLayer(new ConvolutionLayer(24, 24))
        .addLayer(new ConvolutionLayer(24, 24))
        .addLayer(new AdditionBlock()
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new MultiHeadAttentionLayer(2, 100, false).setMask())
        )
        .addLayer(new AdditionBlock()
                .addLayer(new NormalizationLayer2D(false))
                .addLayer(new DenseLayer2D(48, false))
                .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),false))
                .addLayer(new DenseLayer2D(24, false))
        )
        .addLayer(new NormalizationLayer2D(false))
        .addLayer(new FlattenLayer2D(false))
        .addLayer(new DenseLayer(176, false))
        .setOptimizer(optimizer)
        .setFunctionLoss(new FunctionLoss.MSE())
        .setTrainable(true)
        .create();*/

        /*network = new NeuralNetwork();
        network.addInputLayer(24, 480)
        .addLayer(new TYPE2Float2D())
        .addLayer(new VITPositionalEmbeddingLayer(false))
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new MultiHeadAttentionLayer(8, 100, false).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new DenseLayer2D(960, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),false))
            .addLayer(new DenseLayer2D(480, false))
        )
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new MultiHeadAttentionLayer(8, 100, false).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new DenseLayer2D(960, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),false))
            .addLayer(new DenseLayer2D(480, false))
        )
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new MultiHeadAttentionLayer(8, 100, false).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new NormalizationLayer2D(false))
            .addLayer(new DenseLayer2D(960, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),false))
            .addLayer(new DenseLayer2D(480, false))
        )
        .addLayer(new NormalizationLayer2D(false))
        .addLayer(new FlattenLayer2D(false))
        .addLayer(new DenseLayer(176, false))
        .setOptimizer(optimizer)
        .setFunctionLoss(new FunctionLoss.MAE())
        .setTrainable(true)
        .create();*/

        optimizer.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation_optimizer.txt")));

        /*for (int s = 0; s < 150; s++) {
            NNData2D Data2D = loader.getNextTestData(1);
            NNArray[] Output = Data2D.getOutput();
            String TextReal =  Output[0].GetFirstSingleValue(loader, 176);

            String Text = network.query(Data2D.getInput())[0].GetFirstSingleValue(loader, 176);

            byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
            String newstr = new String(charset, StandardCharsets.UTF_8);
            System.out.println(newstr);

            CallGarbageCollector();
        }*/

        /*DateTimeFormatter dtf_ = DateTimeFormatter.ofPattern("yyyy.MM.dd_HH.mm.ss");
        LocalDateTime now_ = LocalDateTime.now();
        System.out.println(dtf_.format(now_));
        network.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf_.format(now_) + "_ka_speech_recognation.txt"));*/

        network.info();//77350
        DataTrainer trainer = new DataTrainer(77350, 77350, loader);

        for (int i = 0; i < 1000; i++) {
            //long start = System.nanoTime();
            trainer.train(network, 10, 1, new DataMetric.Top1());

            network.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt"));
            optimizer.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation_optimizer.txt"));

            if (i % 2 == 0) {
                DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy.MM.dd_HH.mm.ss");
                LocalDateTime now = LocalDateTime.now();
                System.out.println(dtf.format(now));

                network.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf.format(now) + "_ka_speech_recognation.txt"));
                optimizer.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf.format(now) + "_ka_speech_recognation_optimizer.txt"));
            }

            //System.out.println((System.nanoTime() - start) / 1000000);

            for (int s = 0; s < 10; s++) {
                NNData2D Data2D = loader.getNextTrainData(1);
                NNArray[] Output = Data2D.getOutput();
                /*String resultReal = Output[0].GetFirstSingleValue(loader, 176);
                String Text = network.query(Data2D.getInput())[0].GetFirstSingleValue(loader, 176);

                byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
                String newstr = new String(charset, StandardCharsets.UTF_8);
                System.out.println(newstr);*/

                CallGarbageCollector();
            }
        }
    }
}

