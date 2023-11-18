package test.speechtotext;

import data.network_train.NNData2D;
import data.network_train.NNData3D;
import data.nlp.PositionUaLoader;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import trainer.DataMetric;
import trainer.DataTrainer;
import utilities.GPUInit;
import utilities.Use;

import java.io.File;
import java.io.FileWriter;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;

import static neural_network.layers.NeuralLayer.CallGarbageCollector;
import static nnarrays.NNArray.GetFirstSingleValueShortStatic;

public class speechtotext {
    static NeuralNetwork network;

    public static void main(String[] args) throws Exception {
        GPUInit.startup();

        PositionLoader loader = new PositionLoader(45);
        loader.setUseReverse(false);

        Optimizer optimizer = new AdamOptimizer();
        /*network = NeuralNetwork.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt")))
                .setOptimizer(optimizer)
                .setFunctionLoss(new FunctionLoss.MSE())
                .setTrainable(true)
                .create();*/

                network = new NeuralNetwork();
                network.addInputLayer(480, 24, 1)
                .addLayer(new ImagePatchesLayer(12,160))
                .addLayer(new AdditionBlock()
                    .addLayer(new AdditionBlock()
                        .addLayer(new AdditionBlock()
                            .addLayer(new AdditionBlock()
                                .addLayer(new AdditionBlock()
                                    .addLayer(new AdditionBlock()
                                        .addLayer(new MultiHeadAttentionLayer(2, 160).setMask())
                                    )
                                    .addLayer(new NormalizationLayer2D())
                                    .addLayer(new AdditionBlock()
                                        .addLayer(new DenseLayer2D(320))
                                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                                        .addLayer(new DenseLayer2D(160))
                                        //.addLayer(new DropoutLayer2D(0.00001))
                                    )
                                    .addLayer(new NormalizationLayer2D())
                                )
                                .addLayer(new AdditionBlock()
                                    .addLayer(new MultiHeadAttentionLayer(2, 160).setMask())
                                )
                                .addLayer(new NormalizationLayer2D())
                                .addLayer(new AdditionBlock()
                                    .addLayer(new DenseLayer2D(320))
                                    .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                                    .addLayer(new DenseLayer2D(160))
                                    //.addLayer(new DropoutLayer2D(0.00001))
                                )
                                .addLayer(new NormalizationLayer2D())
                            )
                            .addLayer(new AdditionBlock()
                                    .addLayer(new MultiHeadAttentionLayer(2, 160).setMask())
                            )
                            .addLayer(new NormalizationLayer2D())
                            .addLayer(new AdditionBlock()
                                .addLayer(new DenseLayer2D(320))
                                .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                                .addLayer(new DenseLayer2D(160))
                                //.addLayer(new DropoutLayer2D(0.00001))
                            )
                            .addLayer(new NormalizationLayer2D())
                        )
                        .addLayer(new AdditionBlock()
                            .addLayer(new MultiHeadAttentionLayer(2, 160).setMask())
                        )
                        .addLayer(new NormalizationLayer2D())
                        .addLayer(new AdditionBlock()
                            .addLayer(new DenseLayer2D(320))
                            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                            .addLayer(new DenseLayer2D(160))
                            //.addLayer(new DropoutLayer2D(0.00001))
                        )
                        .addLayer(new NormalizationLayer2D())
                    )
                    .addLayer(new AdditionBlock()
                        .addLayer(new MultiHeadAttentionLayer(2, 160).setMask())
                    )
                    .addLayer(new NormalizationLayer2D())
                    .addLayer(new AdditionBlock()
                        .addLayer(new DenseLayer2D(320))
                        .addLayer(new ActivationLayer2D(new FunctionActivation.GELU()))
                        .addLayer(new DenseLayer2D(160))
                        //.addLayer(new DropoutLayer2D(0.00001))
                    )
                    .addLayer(new NormalizationLayer2D())
                )
                .addLayer(new NormalizationLayer2D())
                .addLayer(new FlattenLayer2D())
                .addLayer(new DenseLayer(176))
                //.addLayer(new ActivationLayer(new FunctionActivation.Linear()))
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

        network.info();
        DataTrainer trainer = new DataTrainer(5000, 5000, loader);

        for (int i = 0; i < 1000; i++) {
            //long start = System.nanoTime();
            trainer.train(network, 5, 1, new DataMetric.Top1());

            network.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt"));
            optimizer.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation_optimizer.txt"));

            //System.out.println((System.nanoTime() - start) / 1000000);

            for (int s = 0; s < 10; s++) {
                NNData3D Data3D = loader.getNextTestData(1);
                NNArray[] Output = Data3D.getOutput();
                short[] resultReal = GetFirstSingleValueShortStatic(Output[0].getData_gpu(), 176);
                String TextReal = loader.decodeString(resultReal);

                short[] result = GetFirstSingleValueShortStatic(network.query(Data3D.getInput())[0].getData_gpu(), 176);
                String Text = loader.decodeString(result);

                byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
                String newstr = new String(charset, StandardCharsets.UTF_8);
                System.out.println(newstr);

                CallGarbageCollector();
            }
        }
    }
}

