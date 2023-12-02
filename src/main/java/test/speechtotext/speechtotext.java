package test.speechtotext;

import data.network_train.NNData2D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.FlattenLayer2D;
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
       /* network = NeuralNetwork.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt")))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.MAE())
                .setTrainable(true)
                .create();*/

        /*network = new NeuralNetwork();
        network.addInputLayer(24, 480)
        .addLayer(new Half2Float2D())
        .addLayer(new VITPositionalEmbeddingLayer())
        .addLayer(new AdditionBlock()
           .addLayer(new NormalizationLayer2D())
           .addLayer(new Float2Half2D())
           .addLayer(new MultiHeadAttentionLayer(4, 260, true).setMask())
           .addLayer(new Half2Float2D())
           .addLayer(new NormalizationLayer2D())
        )
        .addLayer(new DenseLayer2D(480, false))
        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
        .addLayer(new FlattenLayer2D())
        .addLayer(new DenseLayer(176))
        .setOptimizer(optimizer)
        .setFunctionLoss(new FunctionLoss.MSE())
        .setTrainable(true)
        .create();*/

        network = new NeuralNetwork();
        network.addInputLayer(24, 480)
        .addLayer(new VITPositionalEmbeddingLayer(true))
        //1////////////////////////////////////////////////////////////////////////////////////
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new MultiHeadAttentionLayer(8, 260, true).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new DenseLayer2D(960, true))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),true))
            .addLayer(new DenseLayer2D(480, true))
        )
        //2////////////////////////////////////////////////////////////////////////////////////
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new MultiHeadAttentionLayer(8, 260, true).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new DenseLayer2D(960, true))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),true))
            .addLayer(new DenseLayer2D(480, true))
        )
        //3////////////////////////////////////////////////////////////////////////////////////
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new MultiHeadAttentionLayer(8, 260, true).setMask())
        )
        .addLayer(new AdditionBlock()
            .addLayer(new Half2Float2D())
            .addLayer(new NormalizationLayer2D())
            .addLayer(new Float2Half2D())
            .addLayer(new DenseLayer2D(960, true))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(),true))
            .addLayer(new DenseLayer2D(480, true))
        )
        ///////////////////////////////////////////////////////////////////////////////////////
        .addLayer(new Half2Float2D())
        .addLayer(new NormalizationLayer2D())
        .addLayer(new FlattenLayer2D(false))
        .addLayer(new DenseLayer(176, false))
        .setOptimizer(optimizer)
        .setFunctionLoss(new FunctionLoss.MSE())
        .setTrainable(true)
        .create();

       //optimizer.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation_optimizer.txt")));

        /*for (int s = 0; s < 150; s++) {
            NNData3D Data3D = loader.getNextTestData(1);
            NNArray[] Output = Data3D.getOutput();
            float[] resultReal =  GetFirstSingleValueFloatStatic(Output[0].getData_gpu(), 176);
            String TextReal = loader.decodeString(resultReal);

            float[] result = GetFirstSingleValueFloatStatic(network.query(Data3D.getInput())[0].getData_gpu(), 176);
            String Text = loader.decodeString(result);

            byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
            String newstr = new String(charset, StandardCharsets.UTF_8);
            System.out.println(newstr);

            CallGarbageCollector();
        }*/

        /*DateTimeFormatter dtf_ = DateTimeFormatter.ofPattern("yyyy.MM.dd_HH.mm.ss");
        LocalDateTime now_ = LocalDateTime.now();
        System.out.println(dtf_.format(now_));
        network.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf_.format(now_) + "_ka_speech_recognation.txt"));*/

        network.info();
        DataTrainer trainer = new DataTrainer(5000, 5000, loader);

        for (int i = 0; i < 1000; i++) {
            //long start = System.nanoTime();
            trainer.train(network, 50, 1, new DataMetric.Top1());

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
                NNData2D Data2D = loader.getNextTestData(1);
                NNArray[] Output = Data2D.getOutput();
                String resultReal = Output[0].GetFirstSingleValue(loader, 176);
                String Text = network.query(Data2D.getInput())[0].GetFirstSingleValue(loader, 176);

                byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
                String newstr = new String(charset, StandardCharsets.UTF_8);
                System.out.println(newstr);

                CallGarbageCollector();
            }
        }
    }
}

