package neural_network.layers.layer_2d;

import jcuda.driver.JCudaDriver;
import lombok.Getter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static utilities.JCudaHelper.CONTEXT;
import static utilities.Use.*;

public class MultiHeadAttentionLayer extends NeuralLayer2D {
    //trainable parts
    private NNMatrix[] weightKey;
    private NNMatrix[] derWeightKey;
    private NNMatrix[] weightQuery;
    private NNMatrix[] derWeightQuery;
    private NNMatrix[] derWeightValue;
    private NNMatrix[] weightValue;
    private NNMatrix weight;
    private NNMatrix derWeight;
    private Regularization regularization;
    private Initializer initializer;
    private boolean loadWeight;

    private final int countHead;
    private final int sizeAttention;
    private final float dropout;

    @Getter
    private boolean useMask;
    private NNMatrix mask;
    private boolean hasEncoderLayer;
    private NeuralLayer encoderLayer;

    private NNMatrix[][] key;
    private NNMatrix[][] query;
    private NNMatrix[][] value;
    private NNMatrix[][] score;
    private NNMatrix[][] inputAtt;
    private NNMatrix[][] outputAtt;
    private NNMatrix[] attention;
    private NNMatrix[] outputDecoder;
    private NNMatrix[] errorDecoder;

    public MultiHeadAttentionLayer(int countHead, int sizeAttention) {
        this(countHead, sizeAttention, 0);
    }

    public MultiHeadAttentionLayer(int countHead, int sizeAttention, double dropout) {
        super();
        this.hasEncoderLayer = false;
        this.useMask = false;
        this.countHead = countHead;
        this.sizeAttention = sizeAttention;
        this.dropout = (float) dropout;
        this.trainable = true;
        initializer = new Initializer.HeNormal();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
        for (int i = 0; i < countHead; i++) {
            optimizer.addDataOptimize(weightKey[i], derWeightKey[i]);
            optimizer.addDataOptimize(weightQuery[i], derWeightQuery[i]);
            optimizer.addDataOptimize(weightValue[i], derWeightValue[i]);
        }
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        depth = size[1];
        width = size[0];
        outWidth = width;
        outDepth = depth;

        if (useMask && mask == null) {
            mask = new NNMatrix(width, width);
            mask.fillUnderDiagonal(1);
        }

        derWeightKey = new NNMatrix[countHead];
        derWeightValue = new NNMatrix[countHead];
        derWeightQuery = new NNMatrix[countHead];

        derWeight = new NNMatrix(countHead * sizeAttention, depth);
        for (int i = 0; i < countHead; i++) {
            derWeightQuery[i] = new NNMatrix(depth, sizeAttention);
            derWeightKey[i] = new NNMatrix(depth, sizeAttention);
            derWeightValue[i] = new NNMatrix(depth, sizeAttention);
        }

        if (!loadWeight) {
            weightKey = new NNMatrix[countHead];
            weightValue = new NNMatrix[countHead];
            weightQuery = new NNMatrix[countHead];

            weight = new NNMatrix(countHead * sizeAttention, depth);
            initializer.initialize(weight);

            for (int i = 0; i < countHead; i++) {
                weightQuery[i] = new NNMatrix(depth, sizeAttention);
                weightKey[i] = new NNMatrix(depth, sizeAttention);
                weightValue[i] = new NNMatrix(depth, sizeAttention);

                initializer.initialize(weightQuery[i]);
                initializer.initialize(weightKey[i]);
                initializer.initialize(weightValue[i]);
            }
        }
    }

    public MultiHeadAttentionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public MultiHeadAttentionLayer setMask(){
       return setMask(null);
    }

    public MultiHeadAttentionLayer addEncoderLayer(NeuralLayer layer){
        this.encoderLayer = layer;
        this.hasEncoderLayer = true;
        layer.addNextLayer(this);
       return this;
    }

    public MultiHeadAttentionLayer setMask(NNMatrix mask){
        this.mask = mask;
        useMask = true;

        return this;
    }

    public MultiHeadAttentionLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;

        return this;
    }

    public MultiHeadAttentionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;

        countParam = weight.size() + weightValue[0].size() * 3 * countHead;
        System.out.println("MultiHeadAtt| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Multi head attention layer\n");
        writer.write(countHead + "\n");
        writer.write(sizeAttention + "\n");
        writer.write(dropout + "\n");
        writer.write(useMask + "\n");
        if(useMask)
        {
            writer.write(mask.getRow() + "\n"); //Daylight
            mask.save(writer);
        }
        weight.save(writer);
        for (int i = 0; i < countHead; i++) {
            weightQuery[i].save(writer);
            weightKey[i].save(writer);
            weightValue[i].save(writer);
        }
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @SneakyThrows
    @Override
    public void generateOutput(NNArray[] inputs) {
        //long start = System.nanoTime();
        this.input = NNArrays.isMatrix(inputs);
        this.output = new NNMatrix[input.length];

        key = new NNMatrix[input.length][];
        query = new NNMatrix[input.length][];
        value = new NNMatrix[input.length][];

        attention = new NNMatrix[input.length];
        score = new NNMatrix[input.length][];
        inputAtt = new NNMatrix[input.length][];
        outputAtt = new NNMatrix[input.length][];

        if (hasEncoderLayer) {
            outputDecoder = NNArrays.isMatrix(encoderLayer.getOutput());
        }

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;

                executor.execute(() -> {
                    output[i] = attention(this.input[i], i);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                output[i] = attention(this.input[i], i);
            }
        }
    }

    private NNMatrix attention(NNMatrix input, int i) {
        key[i] = new NNMatrix[countHead];
        query[i] = new NNMatrix[countHead];
        value[i] = new NNMatrix[countHead];

        score[i] = new NNMatrix[countHead];
        inputAtt[i] = new NNMatrix[countHead];
        outputAtt[i] = new NNMatrix[countHead];

        attention[i] = new NNMatrix(width, countHead * sizeAttention);

        for (int j = 0; j < countHead; j++) {
            if(hasEncoderLayer){
                key[i][j] = outputDecoder[i].dot(weightKey[j]);
                query[i][j] = outputDecoder[i].dot(weightQuery[j]);
            } else {
                key[i][j] = input.dot(weightKey[j]);
                query[i][j] = input.dot(weightQuery[j]);
            }

            value[i][j] = input.dot(weightValue[j]);

            score[i][j] = query[i][j].dotT(key[i][j]);

            score[i][j].div((float) Math.sqrt(sizeAttention));
            if(useMask){
                score[i][j].mask(mask, 0, -1000000000);
            }

            inputAtt[i][j] = new NNMatrix(score[i][j]);

            inputAtt[i][j].softmax(score[i][j]);

            if (dropout != 0) {
                inputAtt[i][j].dropout(inputAtt[i][j], dropout);
            }

            outputAtt[i][j] = inputAtt[i][j].dot(value[i][j]);

            attention[i].addCopy(outputAtt[i][j], j);
        }

        return attention[i].dot(weight);
    }

    private NNMatrix errorAttention(NNMatrix error, NNMatrix input, int i)
    {
        NNMatrix derAttention = error.dotT(weight);
        if (trainable) {
            derWeight.add(attention[i].transpose().dot(error));
        }

        NNMatrix errorInput = new NNMatrix(input);

        for (int j = 0; j < countHead; j++) {
            NNMatrix errorOutAtt = new NNMatrix(outputAtt[i][j]);
            errorOutAtt.addBackCopy(derAttention, j);

            NNMatrix errorInAtt = errorOutAtt.dotT(value[i][j]);
            NNMatrix errorValue = inputAtt[i][j].transpose().dot(errorOutAtt);

            if (dropout != 0) {
                errorInAtt.dropout(errorInAtt, dropout);
            }

            NNMatrix errorScore = new NNMatrix(score[i][j]);
            errorScore.derSoftmax(inputAtt[i][j], errorInAtt);
            errorScore.div((float) Math.sqrt(sizeAttention));

            NNMatrix errorQuery = errorScore.dot(key[i][j]);
            NNMatrix errorKey = query[i][j].transpose().dotT(errorScore).transpose();

            if(hasEncoderLayer){
                errorDecoder[i].add(errorKey.dotT(weightKey[j]));
                errorDecoder[i].add(errorQuery.dotT(weightQuery[j]));
            } else {
                errorInput.add(errorKey.dotT(weightKey[j]));
                errorInput.add(errorQuery.dotT(weightQuery[j]));
            }
            errorInput.add(errorValue.dotT(weightValue[j]));

            if (trainable) {
                NNMatrix inputT = input.transpose();
                derWeightKey[j].add(inputT.dot(errorKey));
                derWeightQuery[j].add(inputT.dot(errorQuery));
                derWeightValue[j].add(inputT.dot(errorValue));
            }
        }

        return errorInput;
    }

    @Override
    public NNArray[] getErrorNL(){
        return errorDecoder;
    }

    @SneakyThrows
    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[errors.length];
        if (hasEncoderLayer) {
            errorDecoder = new NNMatrix[errors.length];
        }

        if (Use.CPU) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(input.length);
            for (int t = 0; t < input.length; t++) {
                final int i = t;

                executor.execute(() -> {
                    if (hasEncoderLayer) {
                        errorDecoder[i] = new NNMatrix(outputDecoder[i]);
                    }
                    error[i] = errorAttention(errorNL[i], input[i], i);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < input.length; i++) {
                if (hasEncoderLayer) {
                    errorDecoder[i] = new NNMatrix(outputDecoder[i]);
                }
                error[i] = errorAttention(errorNL[i], input[i], i);
            }
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);

            for (int i = 0; i < countHead; i++) {
                regularization.regularization(weightValue[i]);
                regularization.regularization(weightKey[i]);
                regularization.regularization(weightQuery[i]);
            }
        }
    }

    public static MultiHeadAttentionLayer read(Scanner scanner) {
        MultiHeadAttentionLayer layer = new MultiHeadAttentionLayer(Integer.parseInt(scanner.nextLine()),
                Integer.parseInt(scanner.nextLine()),
                Double.parseDouble(scanner.nextLine()));

        layer.useMask = Boolean.parseBoolean(scanner.nextLine());

        //daylight
        ////////////////////////////////////////////////
        int[] size = new int[2];

        if(layer.useMask)
        {
            size[0] = Integer.parseInt(scanner.nextLine());
        }

        size[1] = layer.sizeAttention;

        layer.initialize(size);
        ///////////////////////////////////////////////

        if(layer.useMask)
        {
            layer.mask = NNMatrix.read(scanner);
        }

        layer.weight = NNMatrix.read(scanner);
        for (int i = 0; i < layer.countHead; i++) {
            layer.weightQuery[i] = NNMatrix.read(scanner);
            layer.weightKey[i] = NNMatrix.read(scanner);
            layer.weightValue[i] = NNMatrix.read(scanner);
        }

        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}
