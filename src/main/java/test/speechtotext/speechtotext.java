package test.speechtotext;

import data.network_train.NNData2D;
import data.network_train.NNData3D;
import data.network_train.NNData3DMatrix;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.NormalizationLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.recurrent.LSTMLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.layers.reshape.ReshapeLayer2D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.nlp.Transformer;
import neural_network.network.nlp.TransformerVisual;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
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
import static utilities.Use.WordCount;
import static utilities.Use.sizeVocabulary;

public class speechtotext {
    static NeuralNetwork T;
    static TransformerVisual transformer;

    public static void main(String[] args) throws Exception {
        GPUInit.startup();

        PositionLoader loader = new PositionLoader(44);
        loader.setUseReverse(false);

        Optimizer optimizer = new AdamOptimizer(0.00005);//0.00005

        T = NeuralNetwork.read(new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt")))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(optimizer)
                .setTrainable(true)
                .create();

        /*TransformerVisual transformer = new TransformerVisual();
        transformer.addInputLayer(480, 24);
        transformer.addTYPE2Float2DLayer();
        transformer.addLayer(new ConvolutionLayer(180, 8, 1, 4));
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(180, 3, 1, 1))
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new DenseLayer2D(150 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(180, false))
        );
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(180, 3, 1, 1))
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new DenseLayer2D(150 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(180, false))
        );
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(180, 3, 1, 1))
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new DenseLayer2D(150 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(180, false))
        );
        transformer.addFlattenLayer();
        transformer.addDenseLayer1D(WordCount);
        T = transformer.createTransformer();
        T.setFunctionLoss(new FunctionLoss.MSE());
        T.setOptimizer(optimizer);
        T.create();*/

        /*TransformerVisual transformer = new TransformerVisual();
        transformer.addInputLayer(480, 24);
        transformer.addTYPE2Float2DLayer();
        transformer.addLayer(new ConvolutionLayer(150, 8, 1, 4));
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new ConvolutionLayer(100, 3, 1, 1))
            .addLayer(new DenseLayer2D(100 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(150, false))
        );
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new ConvolutionLayer(100, 3, 1, 1))
            .addLayer(new DenseLayer2D(100 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(150, false))
        );
        transformer.addLayer(new AdditionBlock()
            .addLayer(new ConvolutionLayer(150, 3, 1, 1))
            .addLayer(new ConvolutionLayer(100, 3, 1, 1))
            .addLayer(new DenseLayer2D(100 * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(150, false))
        );
        transformer.addFlattenLayer();
        transformer.addDenseLayer1D(WordCount);
        T = transformer.createTransformer();
        T.setFunctionLoss(new FunctionLoss.MSE());
        T.setOptimizer(optimizer);
        T.create();*/


        /*TransformerVisual transformer = new TransformerVisual();
        transformer.addInputLayer(480, 24);
        transformer.addTYPE2Float2DLayer();
        transformer.addLayer(new AdditionBlock()
                .addLayer(new DenseLayer2D(24 * 2, false))
                .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
                .addLayer(new DenseLayer2D(24, false)));
        transformer.addLayer(new NormalizationLayer2D(false));
        transformer.addLayer(new ConvolutionLayer(100, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(90, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(80, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(70, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(80, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(90, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(80, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(90, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(80, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(70, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(80, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(90, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(200, 3, 1, 1));
        transformer.addLayer(new AdditionBlock()
                .addLayer(new DenseLayer2D(200 * 2, false))
                .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
                .addLayer(new DenseLayer2D(200, false)));
        transformer.addLayer(new NormalizationLayer2D(false));
        transformer.addLayer(new DenseLayer2D(200, false));
        transformer.addFlattenLayer();
        transformer.addDenseLayer1D(WordCount);
        T = transformer.createTransformer();
        T.setFunctionLoss(new FunctionLoss.MSE());
        T.setOptimizer(optimizer);
        T.create();*/

        /*TransformerVisual transformer = new TransformerVisual();
        transformer.addInputLayer(480, 24);
        transformer.addTYPE2Float2DLayer();
        transformer.addLayer(new ConvolutionLayer(24, 3, 1, 1));
        transformer.addLayer(new ConvolutionLayer(24, 3, 1, 1));
        transformer.addVITPositionalEmbedding();
        transformer.addEncoderBlock(2, 480,24);
        transformer.addFlattenLayer();
        transformer.addDenseLayer1D(WordCount);
        T = transformer.createTransformer();
        T.setFunctionLoss(new FunctionLoss.MSE());
        T.setOptimizer(optimizer);
        T.create();*/

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

        T.info();//77350
        //decoder.info();//77350
        DataTrainer trainer = new DataTrainer(77350, 77350, loader);

        for (int i = 0; i < 1000; i++) {
            trainer.train(T, 10, 1, 1, new DataMetric.Top1());

            T.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation.txt"));
            optimizer.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/ka_speech_recognation_optimizer.txt"));

            if (i % 2 == 0) {
                DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy.MM.dd_HH.mm.ss");
                LocalDateTime now = LocalDateTime.now();
                System.out.println(dtf.format(now));

                T.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf.format(now) + "_ka_speech_recognation.txt"));
                optimizer.save(new FileWriter("C:/Levani/NeuralNetworkCPU/data/"+dtf.format(now) + "_ka_speech_recognation_optimizer.txt"));
            }

            //System.out.println((System.nanoTime() - start) / 1000000);

            for (int s = 0; s < 10; s++) {
                NNData2D Data2D = loader.getNextTrainData(1);
                //NNMatrix[] Output = NNArrays.toHotVector(Data2D.getOutput(), sizeVocabulary);
                String resultReal = Data2D.getOutput()[0].GetFirstSingleValue(loader, WordCount/* * sizeVocabulary*/);
                String Text = T.query(Data2D.getInput())[0].GetFirstSingleValue(loader, WordCount/* * sizeVocabulary*/);

                byte[] charset = Text.getBytes(StandardCharsets.UTF_8);
                String newstr = new String(charset, StandardCharsets.UTF_8);
                System.out.println(newstr);

                CallGarbageCollector();
            }
        }
    }
}

