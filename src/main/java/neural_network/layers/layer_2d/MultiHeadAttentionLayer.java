package neural_network.layers.layer_2d;

import lombok.Getter;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MultiHeadAttentionLayer extends NeuralLayer2D {
    //trainable parts
    private NNMatrix[] weightKey;
    private NNMatrix[] derWeightKey;
    private NNMatrix[] weightQuery;
    private NNMatrix[] derWeightQuery;
    private NNMatrix[] weightValue;
    private NNMatrix[] derWeightValue;

    private CublasUtil.Matrix[] weightKey_gpu;
    private CublasUtil.Matrix[] derWeightKey_gpu;
    private CublasUtil.Matrix[] weightQuery_gpu;
    private CublasUtil.Matrix[] derWeightQuery_gpu;
    private CublasUtil.Matrix[] weightValue_gpu;
    private CublasUtil.Matrix[] derWeightValue_gpu;

    private NNMatrix weight;
    private NNMatrix derWeight;

    private CublasUtil.Matrix weight_gpu;
    private CublasUtil.Matrix derWeight_gpu;

    private Regularization regularization;
    private Initializer initializer;
    private boolean loadWeight;

    private final int countHead;
    private final int sizeAttention;
    private final float dropout;

    @Getter
    private boolean useMask;
    private NNMatrix mask;
    private CublasUtil.Matrix mask_gpu;

    private boolean hasEncoderLayer;
    private NeuralLayer encoderLayer;

    private NNMatrix[][] key;
    private NNMatrix[][] query;
    private NNMatrix[][] value;

    private CublasUtil.Matrix[][] key_gpu;
    private CublasUtil.Matrix[][] query_gpu;
    private CublasUtil.Matrix[][] value_gpu;

    private NNMatrix[][] score;
    private CublasUtil.Matrix[][] score_gpu;
    private NNMatrix[][] inputAtt;
    private CublasUtil.Matrix[][] inputAtt_gpu;
    private NNMatrix[][] outputAtt;
    private CublasUtil.Matrix[][] outputAtt_gpu;
    private NNMatrix[] attention;
    private CublasUtil.Matrix[] attention_gpu;
    private NNMatrix[] outputDecoder;
    private CublasUtil.Matrix[] outputDecoder_gpu;
    private NNMatrix[] errorDecoder;
    private CublasUtil.Matrix[] errorDecoder_gpu;

    public boolean UseGPU = true;
    public boolean UseCPU = false;

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
        if (UseCPU)
        {
            optimizer.addDataOptimize(weight, derWeight);
            for (int i = 0; i < countHead; i++) {
                optimizer.addDataOptimize(weightKey[i], derWeightKey[i]);
                optimizer.addDataOptimize(weightQuery[i], derWeightQuery[i]);
                optimizer.addDataOptimize(weightValue[i], derWeightValue[i]);
            }
        }

        if (UseGPU)
        {
            optimizer.addDataOptimize(weight_gpu, derWeight_gpu);
            for (int i = 0; i < countHead; i++) {
                optimizer.addDataOptimize(weightKey_gpu[i], derWeightKey_gpu[i]);
                optimizer.addDataOptimize(weightQuery_gpu[i], derWeightQuery_gpu[i]);
                optimizer.addDataOptimize(weightValue_gpu[i], derWeightValue_gpu[i]);
            }
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

        if(useMask && mask == null){
            if (UseCPU)
            {
                mask = new NNMatrix(width, width);
                mask.fillUnderDiagonal(1);
            }

            if (UseGPU)
            {
                mask_gpu = new CublasUtil.Matrix(width, width);
                mask_gpu = mask_gpu.diagAddi(1);
            }
        }

        if (UseCPU)
        {
            derWeightKey = new NNMatrix[countHead];
            derWeightValue = new NNMatrix[countHead];
            derWeightQuery = new NNMatrix[countHead];
        }

        if (UseGPU)
        {
            derWeightKey_gpu = new CublasUtil.Matrix[countHead];
            derWeightValue_gpu = new CublasUtil.Matrix[countHead];
            derWeightQuery_gpu = new CublasUtil.Matrix[countHead];
        }

        if (UseCPU)
        {
            derWeight = new NNMatrix(countHead * sizeAttention, depth);
            for (int i = 0; i < countHead; i++) {
                derWeightQuery[i] = new NNMatrix(depth, sizeAttention);
                derWeightKey[i] = new NNMatrix(depth, sizeAttention);
                derWeightValue[i] = new NNMatrix(depth, sizeAttention);
            }
        }

        if (UseGPU)
        {
            derWeight_gpu = new CublasUtil.Matrix(countHead * sizeAttention, depth);
            for (int i = 0; i < countHead; i++) {
                derWeightQuery_gpu[i] = new CublasUtil.Matrix(depth, sizeAttention);
                derWeightKey_gpu[i] = new CublasUtil.Matrix(depth, sizeAttention);
                derWeightValue_gpu[i] = new CublasUtil.Matrix(depth, sizeAttention);
            }
        }

        if (!loadWeight) {
            if (UseCPU)
            {
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

            if (UseGPU) {
                weightKey_gpu = new CublasUtil.Matrix[countHead];
                weightValue_gpu = new CublasUtil.Matrix[countHead];
                weightQuery_gpu = new CublasUtil.Matrix[countHead];

                Random rand = new Random(1);
                if ((UseCPU) && (UseGPU)) {
                    weight_gpu = CublasUtil.Matrix.build(weight.getRow(), weight.getColumn(), weight.getData());
                } else {
                    weight_gpu = CublasUtil.Matrix.rand(countHead * sizeAttention, depth, rand);
                }

                for (int i = 0; i < countHead; i++) {
                    if ((UseCPU) && (UseGPU)) {
                        weightQuery_gpu[i] = CublasUtil.Matrix.build(weightQuery[i].getRow(), weightQuery[i].getColumn(), weightQuery[i].getData());
                        weightKey_gpu[i] = CublasUtil.Matrix.build(weightKey[i].getRow(), weightKey[i].getColumn(), weightKey[i].getData());
                        weightValue_gpu[i] = CublasUtil.Matrix.build(weightValue[i].getRow(), weightValue[i].getColumn(), weightValue[i].getData());
                    }
                    else {
                        weightQuery_gpu[i] = CublasUtil.Matrix.rand(sizeAttention, depth, rand);
                        weightKey_gpu[i] = CublasUtil.Matrix.rand(sizeAttention, depth, rand);
                        weightValue_gpu[i] = CublasUtil.Matrix.rand(sizeAttention, depth, rand);
                    }
                }
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

        if (UseCPU) {
            countParam = weight.size() + weightValue[0].size() * 3 * countHead;
            System.out.println("MultiHeadAtt| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        }

        if (UseGPU) {
            countParam = weight_gpu.rows() * weight_gpu.cols() + weightValue_gpu[0].rows() * weightValue_gpu[0].cols() * 3 * countHead;
            System.out.println("MultiHeadAtt| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        }

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
        this.input = NNArrays.isMatrix(inputs);
        this.output = new NNMatrix[input.length];

        if (UseCPU)
        {
            key = new NNMatrix[input.length][];
            query = new NNMatrix[input.length][];
            value = new NNMatrix[input.length][];

            attention = new NNMatrix[input.length];
            score = new NNMatrix[input.length][];
            inputAtt = new NNMatrix[input.length][];
            outputAtt = new NNMatrix[input.length][];
        }

        if (UseGPU)
        {
            key_gpu = new CublasUtil.Matrix[input.length][];
            query_gpu = new CublasUtil.Matrix[input.length][];
            value_gpu = new CublasUtil.Matrix[input.length][];

            attention_gpu = new CublasUtil.Matrix[input.length];
            score_gpu = new CublasUtil.Matrix[input.length][];
            inputAtt_gpu = new CublasUtil.Matrix[input.length][];
            outputAtt_gpu = new CublasUtil.Matrix[input.length][];
        }

        if (hasEncoderLayer) {
            if (UseCPU) {
                outputDecoder = NNArrays.isMatrix(encoderLayer.getOutput());
            }

            if (UseGPU) {
                NNMatrix[] ArrayMatrix = NNArrays.isMatrix(encoderLayer.getOutput());
                for (int i = 0; i < ArrayMatrix.length; i++)
                    outputDecoder_gpu[i] = CublasUtil.Matrix.build(ArrayMatrix[i].getRow(), ArrayMatrix[i].getColumn(), ArrayMatrix[i].getData());
            }
        }

        if ((UseCPU) && (!UseGPU)) {
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
        }

        if (UseGPU) {
            for (int i = 0; i < input.length; i++) {
                output[i] = attention(this.input[i], i);
            }

            if (hasEncoderLayer) {
                for (int i = 0; i < outputDecoder_gpu.length; i++) {
                    outputDecoder_gpu[i].free();
                }
            }
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] inputs) {
        input_gpu = inputs;
        this.output = new NNMatrix[input_gpu.length];

        key_gpu = new CublasUtil.Matrix[input_gpu.length][];
        query_gpu = new CublasUtil.Matrix[input_gpu.length][];
        value_gpu = new CublasUtil.Matrix[input_gpu.length][];

        attention_gpu = new CublasUtil.Matrix[input_gpu.length];
        score_gpu = new CublasUtil.Matrix[input_gpu.length][];
        inputAtt_gpu = new CublasUtil.Matrix[input_gpu.length][];
        outputAtt_gpu = new CublasUtil.Matrix[input_gpu.length][];

        if (hasEncoderLayer) {
            outputDecoder_gpu = encoderLayer.getOutput_gpu();
        }

        for (int i = 0; i < input_gpu.length; i++) {
            output[i] = attention(input_gpu[i], i);
        }

        if (hasEncoderLayer) {
            for (int i = 0; i < outputDecoder_gpu.length; i++) {
                outputDecoder_gpu[i].free();
            }
        }
    }

    private NNMatrix attention(CublasUtil.Matrix input, int i) {
        key_gpu[i] = new CublasUtil.Matrix[countHead];
        query_gpu[i] = new CublasUtil.Matrix[countHead];
        value_gpu[i] = new CublasUtil.Matrix[countHead];

        score_gpu[i] = new CublasUtil.Matrix[countHead];
        inputAtt_gpu[i] = new CublasUtil.Matrix[countHead];
        outputAtt_gpu[i] = new CublasUtil.Matrix[countHead];

        attention_gpu[i] = new CublasUtil.Matrix(width, countHead * sizeAttention);

        for (int j = 0; j < countHead; j++) {
            if (hasEncoderLayer) {
                key_gpu[i][j] = outputDecoder_gpu[i].dot(weightKey_gpu[i]);
                query_gpu[i][j] = outputDecoder_gpu[i].dot(weightQuery_gpu[i]);
            } else {
                key_gpu[i][j] = input.dot(weightKey_gpu[j]);
                query_gpu[i][j] = input.dot(weightQuery_gpu[j]);
            }

            value_gpu[i][j] = input.dot(weightValue_gpu[j]);

            score_gpu[i][j] = query_gpu[i][j].dotT(key_gpu[i][j]);

            score_gpu[i][j].div((float) Math.sqrt(sizeAttention));

            if (useMask) {
                score_gpu[i][j].mask(0, -1000000000, mask_gpu);
            }

            inputAtt_gpu[i][j] = new CublasUtil.Matrix(score_gpu[i][j].rows(), score_gpu[i][j].cols());
            inputAtt_gpu[i][j].softmax(score_gpu[i][j]);

            if (dropout != 0) {
                inputAtt_gpu[i][j].dropout((float) Math.random(), dropout);
            }

            outputAtt_gpu[i][j] = inputAtt_gpu[i][j].dot(value_gpu[i][j]);
            attention_gpu[i].addCopy(outputAtt_gpu[i][j], j);
        }

        input.free();

        CublasUtil.Matrix C_matrix = attention_gpu[i].dot(weight_gpu);
        NNMatrix result = new NNMatrix(C_matrix.rows(), C_matrix.cols(), C_matrix.toArray());

        C_matrix.free();

        return result;
    }

    private NNMatrix attention(NNMatrix input, int i) {
        if (UseCPU) {
            key[i] = new NNMatrix[countHead];
            query[i] = new NNMatrix[countHead];
            value[i] = new NNMatrix[countHead];

            score[i] = new NNMatrix[countHead];
            inputAtt[i] = new NNMatrix[countHead];
            outputAtt[i] = new NNMatrix[countHead];

            attention[i] = new NNMatrix(width, countHead * sizeAttention);

            for (int j = 0; j < countHead; j++) {
                if (hasEncoderLayer) {
                    key[i][j] = outputDecoder[i].dot(weightKey[j]);
                    query[i][j] = outputDecoder[i].dot(weightQuery[j]);
                } else {
                    key[i][j] = input.dot(weightKey[j]);
                    query[i][j] = input.dot(weightQuery[j]);
                }

                value[i][j] = input.dot(weightValue[j]);

                score[i][j] = query[i][j].dotT(key[i][j]);

                score[i][j].div((float) Math.sqrt(sizeAttention));
                if (useMask) {
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

            if (!((UseCPU) && (UseGPU))) {
                return attention[i].dot(weight);
            }
        }

        if (UseGPU) {
            key_gpu[i] = new CublasUtil.Matrix[countHead];
            query_gpu[i] = new CublasUtil.Matrix[countHead];
            value_gpu[i] = new CublasUtil.Matrix[countHead];

            score_gpu[i] = new CublasUtil.Matrix[countHead];
            inputAtt_gpu[i] = new CublasUtil.Matrix[countHead];
            outputAtt_gpu[i] = new CublasUtil.Matrix[countHead];

            attention_gpu[i] = new CublasUtil.Matrix(width, countHead * sizeAttention);

            CublasUtil.Matrix input_gpu = CublasUtil.Matrix.build(input.getRow(), input.getColumn(), input.getData());

            for (int j = 0; j < countHead; j++) {
                if (hasEncoderLayer) {
                    key_gpu[i][j] = outputDecoder_gpu[i].dot(weightKey_gpu[i]);
                    query_gpu[i][j] = outputDecoder_gpu[i].dot(weightQuery_gpu[i]);
                } else {
                    key_gpu[i][j] = input_gpu.dot(weightKey_gpu[j]);
                    query_gpu[i][j] = input_gpu.dot(weightQuery_gpu[j]);
                }

                value_gpu[i][j] = input_gpu.dot(weightValue_gpu[j]);

                score_gpu[i][j] = query_gpu[i][j].dotT(key_gpu[i][j]);

                score_gpu[i][j].div((float) Math.sqrt(sizeAttention));

                if (useMask) {
                    score_gpu[i][j].mask(0, -1000000000, mask_gpu);
                }

                inputAtt_gpu[i][j] = new CublasUtil.Matrix(score_gpu[i][j].rows(), score_gpu[i][j].cols());
                inputAtt_gpu[i][j].softmax(score_gpu[i][j]);

                if (dropout != 0) {
                    inputAtt_gpu[i][j].dropout((float) Math.random(), dropout);
                }

                outputAtt_gpu[i][j] = inputAtt_gpu[i][j].dot(value_gpu[i][j]);
                attention_gpu[i].addCopy(outputAtt_gpu[i][j], j);
            }

            input_gpu.free();

            CublasUtil.Matrix C_matrix = attention_gpu[i].dot(weight_gpu);
            NNMatrix result = new NNMatrix(C_matrix.rows(), C_matrix.cols(), C_matrix.toArray());

            C_matrix.free();

            return result;
        }
        return null;
    }

    private NNMatrix errorAttention(NNMatrix error, NNMatrix input, int i)
    {
        NNMatrix derAttention = null;
        CublasUtil.Matrix derAttention_gpu = null;

        NNMatrix errorInput = null;
        CublasUtil.Matrix errorInput_gpu = null;

        CublasUtil.Matrix input_gpu = null;

        if (UseCPU)
        {
            derAttention = error.dotT(weight);
            if (trainable) {
                derWeight.add(attention[i].transpose().dot(error));
            }

            errorInput = new NNMatrix(input);
        }

        if (UseGPU)
        {
            CublasUtil.Matrix error_gpu = CublasUtil.Matrix.build(error.getRow(), error.getColumn(), error.getData());

            derAttention_gpu = error_gpu.dotT(weight_gpu);
            if (trainable) {
                CublasUtil.Matrix transpose = attention_gpu[i].transpose();
                CublasUtil.Matrix temp = transpose.dot(error_gpu);
                derWeight_gpu.add_(temp);
                temp.free();
                transpose.free();
            }

            errorInput_gpu = CublasUtil.Matrix.build(input.getRow(), input.getColumn(), input.getData());
            input_gpu = CublasUtil.Matrix.build(input.getRow(), input.getColumn(), input.getData());

            error_gpu.free();
        }

        for (int j = 0; j < countHead; j++) {
            if (UseCPU) {
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

                if (hasEncoderLayer) {
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

            if (UseGPU)
            {
                CublasUtil.Matrix errorOutAtt_gpu = new CublasUtil.Matrix(outputAtt_gpu[i][j].rows(), outputAtt_gpu[i][j].cols());
                errorOutAtt_gpu.addBackCopy(derAttention_gpu, j);

                CublasUtil.Matrix errorInAtt_gpu = errorOutAtt_gpu.dotT(value_gpu[i][j]);
                CublasUtil.Matrix transpose = inputAtt_gpu[i][j].transpose();
                CublasUtil.Matrix errorValue_gpu = transpose.dot(errorOutAtt_gpu);
                transpose.free();
                errorOutAtt_gpu.free();

                if (dropout != 0) {
                    errorInAtt_gpu.dropout((float)Math.random(), dropout);
                }

                CublasUtil.Matrix errorScore_gpu = new CublasUtil.Matrix(score_gpu[i][j].rows(), score_gpu[i][j].cols());
                errorScore_gpu.derSoftmax(inputAtt_gpu[i][j], errorInAtt_gpu);
                errorScore_gpu.div((float) Math.sqrt(sizeAttention));
                errorInAtt_gpu.free();

                CublasUtil.Matrix errorQuery_gpu = errorScore_gpu.dot(key_gpu[i][j]);
                CublasUtil.Matrix ta = query_gpu[i][j].transpose();
                CublasUtil.Matrix tt = ta.dotT(errorScore_gpu);
                ta.free();
                CublasUtil.Matrix errorKey_gpu = tt.transpose();
                tt.free();
                errorScore_gpu.free();

                if(hasEncoderLayer){
                    CublasUtil.Matrix temp = errorKey_gpu.dotT(weightKey_gpu[j]);
                    errorDecoder_gpu[i].add_(temp);
                    temp.free();
                    CublasUtil.Matrix temp2 = errorQuery_gpu.dotT(weightQuery_gpu[j]);
                    errorDecoder_gpu[i].add_(temp2);
                    temp2.free();
                } else {
                    CublasUtil.Matrix temp = errorKey_gpu.dotT(weightKey_gpu[j]);
                    errorInput_gpu.add_(temp);
                    temp.free();
                    CublasUtil.Matrix temp2 = errorQuery_gpu.dotT(weightQuery_gpu[j]);
                    errorInput_gpu.add_(temp2);
                    temp2.free();
                }
                CublasUtil.Matrix temp = errorValue_gpu.dotT(weightValue_gpu[j]);
                errorInput_gpu.add_(temp);
                temp.free();

                if (trainable) {
                    CublasUtil.Matrix inputT = input_gpu.transpose();

                    CublasUtil.Matrix temp2 = inputT.dot(errorKey_gpu);
                    derWeightKey_gpu[j].add_(temp2);
                    temp2.free();

                    CublasUtil.Matrix temp3 = inputT.dot(errorQuery_gpu);
                    derWeightQuery_gpu[j].add_(temp3);
                    temp3.free();

                    CublasUtil.Matrix temp4 = inputT.dot(errorValue_gpu);
                    derWeightValue_gpu[j].add_(temp4);
                    temp4.free();

                    inputT.free();
                }
                errorKey_gpu.free();
                errorQuery_gpu.free();
                errorValue_gpu.free();
            }
        }

        if (UseGPU) {
            errorInput = new NNMatrix(errorInput_gpu.rows(), errorInput_gpu.cols(), errorInput_gpu.toArray());
            errorInput_gpu.free();
            input_gpu.free();
            derAttention_gpu.free();

            for (int j = 0; j < countHead; j++) {
                inputAtt_gpu[i][j].free();
                outputAtt_gpu[i][j].free();
                score_gpu[i][j].free();

                key_gpu[i][j].free();
                value_gpu[i][j].free();
                query_gpu[i][j].free();
            }
            attention_gpu[i].free();
        }

        return errorInput;
    }

    private CublasUtil.Matrix errorAttention(CublasUtil.Matrix error_gpu, CublasUtil.Matrix input, int i)
    {
        CublasUtil.Matrix derAttention_gpu = null;
        CublasUtil.Matrix errorInput_gpu = null;
        CublasUtil.Matrix input_gpu = null;

        derAttention_gpu = error_gpu.dotT(weight_gpu);
        if (trainable) {
            CublasUtil.Matrix transpose = attention_gpu[i].transpose();
            CublasUtil.Matrix temp = transpose.dot(error_gpu);
            derWeight_gpu.add_(temp);
            temp.free();
            transpose.free();
        }

        for (int j = 0; j < countHead; j++) {
            CublasUtil.Matrix errorOutAtt_gpu = new CublasUtil.Matrix(outputAtt_gpu[i][j].rows(), outputAtt_gpu[i][j].cols());
            errorOutAtt_gpu.addBackCopy(derAttention_gpu, j);

            CublasUtil.Matrix errorInAtt_gpu = errorOutAtt_gpu.dotT(value_gpu[i][j]);
            CublasUtil.Matrix transpose = inputAtt_gpu[i][j].transpose();
            CublasUtil.Matrix errorValue_gpu = transpose.dot(errorOutAtt_gpu);
            transpose.free();
            errorOutAtt_gpu.free();

            if (dropout != 0) {
                errorInAtt_gpu.dropout((float) Math.random(), dropout);
            }

            CublasUtil.Matrix errorScore_gpu = new CublasUtil.Matrix(score_gpu[i][j].rows(), score_gpu[i][j].cols());
            errorScore_gpu.derSoftmax(inputAtt_gpu[i][j], errorInAtt_gpu);
            errorScore_gpu.div((float) Math.sqrt(sizeAttention));
            errorInAtt_gpu.free();

            CublasUtil.Matrix errorQuery_gpu = errorScore_gpu.dot(key_gpu[i][j]);
            CublasUtil.Matrix ta = query_gpu[i][j].transpose();
            CublasUtil.Matrix tt = ta.dotT(errorScore_gpu);
            ta.free();
            CublasUtil.Matrix errorKey_gpu = tt.transpose();
            tt.free();
            errorScore_gpu.free();

            if (hasEncoderLayer) {
                CublasUtil.Matrix temp = errorKey_gpu.dotT(weightKey_gpu[j]);
                errorDecoder_gpu[i].add_(temp);
                temp.free();
                CublasUtil.Matrix temp2 = errorQuery_gpu.dotT(weightQuery_gpu[j]);
                errorDecoder_gpu[i].add_(temp2);
                temp2.free();
            } else {
                CublasUtil.Matrix temp = errorKey_gpu.dotT(weightKey_gpu[j]);
                errorInput_gpu.add_(temp);
                temp.free();
                CublasUtil.Matrix temp2 = errorQuery_gpu.dotT(weightQuery_gpu[j]);
                errorInput_gpu.add_(temp2);
                temp2.free();
            }
            CublasUtil.Matrix temp = errorValue_gpu.dotT(weightValue_gpu[j]);
            errorInput_gpu.add_(temp);
            temp.free();

            if (trainable) {
                CublasUtil.Matrix inputT = input_gpu.transpose();

                CublasUtil.Matrix temp2 = inputT.dot(errorKey_gpu);
                derWeightKey_gpu[j].add_(temp2);
                temp2.free();

                CublasUtil.Matrix temp3 = inputT.dot(errorQuery_gpu);
                derWeightQuery_gpu[j].add_(temp3);
                temp3.free();

                CublasUtil.Matrix temp4 = inputT.dot(errorValue_gpu);
                derWeightValue_gpu[j].add_(temp4);
                temp4.free();

                inputT.free();
            }
            errorKey_gpu.free();
            errorQuery_gpu.free();
            errorValue_gpu.free();
        }

        derAttention_gpu.free();

        for (int j = 0; j < countHead; j++) {
            inputAtt_gpu[i][j].free();
            outputAtt_gpu[i][j].free();
            score_gpu[i][j].free();

            key_gpu[i][j].free();
            value_gpu[i][j].free();
            query_gpu[i][j].free();
        }
        attention_gpu[i].free();

        return errorInput_gpu;
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
            if (UseCPU) {
                errorDecoder = new NNMatrix[errors.length];
            }

            if (UseGPU) {
                {
                    errorDecoder_gpu = new CublasUtil.Matrix[errors.length];
                }
            }
        }

        if ((UseCPU) && (!UseGPU)) {
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
        }

        if (UseGPU) {
            for (int i = 0; i < input.length; i++) {
                if (hasEncoderLayer) {
                    errorDecoder_gpu[i] = new CublasUtil.Matrix(outputDecoder[i].getRow(), outputDecoder[i].getColumn());
                }
                error[i] = errorAttention(errorNL[i], input[i], i);

                if (hasEncoderLayer) {
                    errorDecoder_gpu[i].free();
                }
            }
        }

        if (trainable && regularization != null) {
            if (UseCPU) {
                regularization.regularization(weight);

                for (int i = 0; i < countHead; i++) {
                    regularization.regularization(weightValue[i]);
                    regularization.regularization(weightKey[i]);
                    regularization.regularization(weightQuery[i]);
                }
            }

            if (UseGPU) {
                /*regularization.regularization(weight_gpu);

                for (int i = 0; i < countHead; i++) {
                    regularization.regularization(weightValue_gpu[i]);
                    regularization.regularization(weightKey_gpu[i]);
                    regularization.regularization(weightQuery_gpu[i]);
                }*/
            }
        }
    }

    @SneakyThrows
    @Override
    public void generateError(CublasUtil.Matrix[] errors) {
        errorNL_gpu = getErrorNextLayer(errors);
        this.error = new NNMatrix[errors.length];

        if (hasEncoderLayer) {
            errorDecoder_gpu = new CublasUtil.Matrix[errors.length];
        }

        for (int i = 0; i < input_gpu.length; i++) {
            if (hasEncoderLayer) {
                errorDecoder_gpu[i] = new CublasUtil.Matrix(outputDecoder[i].getRow(), outputDecoder[i].getColumn());
            }
            error_gpu[i] = errorAttention(errorNL_gpu[i], input_gpu[i], i);

            if (hasEncoderLayer) {
                errorDecoder_gpu[i].free();
            }
        }

        if (trainable && regularization != null) {
            /*regularization.regularization(weight_gpu);

            for (int i = 0; i < countHead; i++) {
                regularization.regularization(weightValue_gpu[i]);
                regularization.regularization(weightKey_gpu[i]);
                regularization.regularization(weightQuery_gpu[i]);
            }*/
        }
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
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
