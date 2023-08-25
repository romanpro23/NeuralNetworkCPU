package neural_network.layers.recurrent;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GRULuongAttentionLayer extends LuongAttentionLayer {
    private NNVector[][] hidden;
    private NNVector[][] resetHidden;
    private NNVector[][] updateGateInput;
    private NNVector[][] updateGateOutput;
    private NNVector[][] resetGateInput;
    private NNVector[][] resetGateOutput;
    protected NNVector[][] inputHidden;
    protected NNVector[][] outputHidden;

    private NNMatrix[] weightInput;
    private NNMatrix[] derWeightInput;

    private NNMatrix[] weightHidden;
    private NNMatrix[] derWeightHidden;

    private NNVector[] threshold;
    private NNVector[] derThreshold;

    private final FunctionActivation functionActivationSigmoid;
    private final FunctionActivation functionActivationTanh;

    public GRULuongAttentionLayer(int countNeuron) {
        this(countNeuron, 0, Attention.DOT);
    }

    public GRULuongAttentionLayer(GRULuongAttentionLayer layer) {
        this(layer.countNeuron, layer.recurrentDropout, layer.attention, layer.returnSequences);
        this.copy(layer);
    }

    public GRULuongAttentionLayer(int countNeuron, double recurrentDropout, Attention attention) {
        super(countNeuron, recurrentDropout, attention);

        this.functionActivationTanh = new FunctionActivation.Tanh();
        this.functionActivationSigmoid = new FunctionActivation.Sigmoid();
    }

    public GRULuongAttentionLayer(int countNeuron, double recurrentDropout, Attention attention, boolean returnSequences) {
        this(countNeuron, recurrentDropout, attention);
        setReturnSequences(returnSequences);
    }

    public GRULuongAttentionLayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;

        return this;
    }

    public GRULuongAttentionLayer setEncoderLayer(RecurrentNeuralLayer layer) {
        super.setEncoderLayer(layer);

        return this;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);
        derThreshold = new NNVector[3];
        derWeightInput = new NNMatrix[3];
        derWeightHidden = new NNMatrix[3];

        for (int i = 0; i < 3; i++) {
            derThreshold[i] = new NNVector(countNeuron);
            derWeightInput[i] = new NNMatrix(countNeuron, depth);
            derWeightHidden[i] = new NNMatrix(countNeuron, countNeuron);
        }

        if (!loadWeight) {
            threshold = new NNVector[3];
            weightInput = new NNMatrix[3];
            weightHidden = new NNMatrix[3];

            for (int i = 0; i < 3; i++) {
                threshold[i] = new NNVector(countNeuron);
                weightInput[i] = new NNMatrix(countNeuron, depth);
                weightHidden[i] = new NNMatrix(countNeuron, countNeuron);

                initializerInput.initialize(weightInput[i]);
                initializerHidden.initialize(weightHidden[i]);
            }

            if (attention == Attention.GENERAL) {
                initializerInput.initialize(weightVal);
            } else if (attention == Attention.CONCAT) {
                initializerInput.initialize(weightAttention);
                initializerInput.initialize(weightVal);
            }
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void initialize(Optimizer optimizer) {
        super.initialize(optimizer);
        for (int i = 0; i < 3; i++) {
            optimizer.addDataOptimize(weightInput[i], derWeightInput[i]);
            optimizer.addDataOptimize(weightHidden[i], derWeightHidden[i]);
            optimizer.addDataOptimize(threshold[i], derThreshold[i]);
        }
    }

    @Override
    public void generateError(CublasUtil.Matrix[] errors) {

    }

    @Override
    public int info() {
        int countParam = (weightHidden[0].size() + weightInput[0].size() + threshold[0].size()) * 3;
        if (attention == Attention.GENERAL) {
            countParam += weightVal.size();
        } else if (attention == Attention.CONCAT) {
            countParam += weightVal.size();
            countParam += weightAttention.size();
        }

        System.out.println("Luong GRU\t| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("GRU luong attention layer\n");
        writer.write(countNeuron + "\n");
        writer.write(recurrentDropout + "\n");
        writer.write(returnSequences + "\n");
        writeAttention(writer);

        for (int i = 0; i < 3; i++) {
            threshold[i].save(writer);
            weightInput[i].save(writer);
            weightHidden[i].save(writer);
        }

        if (attention == Attention.GENERAL) {
            weightVal.save(writer);
        } else if (attention == Attention.CONCAT) {
            weightVal.save(writer);
            weightAttention.save(writer);
        }

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public void generateOutput(NNArray[] inputs, NNArray[][] state) {
        this.input = NNArrays.isMatrix(inputs);
        this.output = new NNMatrix[inputs.length];
        this.inputHidden = new NNVector[inputs.length][];
        this.outputHidden = new NNVector[inputs.length][];

        this.hidden = new NNVector[inputs.length][];
        this.resetHidden = new NNVector[inputs.length][];
        this.resetGateInput = new NNVector[inputs.length][];
        this.resetGateOutput = new NNVector[inputs.length][];
        this.updateGateInput = new NNVector[inputs.length][];
        this.updateGateOutput = new NNVector[inputs.length][];
        this.state = new NNVector[input.length][1];
        this.inputState = new NNVector[input.length][];

        outputPreLayer = NNArrays.isMatrix(encoderLayer.getOutput());
        initializeMemory(inputs.length);

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int cor = 0; cor < inputs.length; cor++) {
            final int i = cor;
            executor.execute(() -> {
                if (state != null) {
                    inputState[i] = NNArrays.isVector(state[i]);
                } else {
                    inputState[i] = null;
                }
                generateOutput(i, inputState[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void generateOutput(int i, NNVector[] states) {
        int countRow = (returnSequences) ? input[i].getRow() : 1;
        output[i] = new NNMatrix(countRow, countNeuron * 2);

        inputHidden[i] = new NNVector[input[i].getRow()];
        outputHidden[i] = new NNVector[input[i].getRow()];

        this.hidden[i] = new NNVector[input[i].getRow()];
        this.resetHidden[i] = new NNVector[input[i].getRow()];
        this.resetGateInput[i] = new NNVector[input[i].getRow()];
        this.resetGateOutput[i] = new NNVector[input[i].getRow()];
        this.updateGateInput[i] = new NNVector[input[i].getRow()];
        this.updateGateOutput[i] = new NNVector[input[i].getRow()];

        initializeMemory(i, input[i].getRow());

        //pass through time
        for (int t = 0, tOut = 0; t < input[i].getRow(); t++) {
            inputHidden[i][t] = new NNVector(countNeuron);
            outputHidden[i][t] = new NNVector(countNeuron);

            this.hidden[i][t] = new NNVector(countNeuron);
            this.resetHidden[i][t] = new NNVector(countNeuron);
            this.resetGateInput[i][t] = new NNVector(countNeuron);
            this.resetGateOutput[i][t] = new NNVector(countNeuron);
            this.updateGateInput[i][t] = new NNVector(countNeuron);
            this.updateGateOutput[i][t] = new NNVector(countNeuron);

            NNVector hidden_t = null;
            if (t > 0) {
                hidden_t = hidden[i][t - 1];
            } else if (states != null) {
                hidden_t = states[0];
            }

            //generate new hidden state for update and reset gate
            updateGateInput[i][t].set(threshold[0]);
            resetGateInput[i][t].set(threshold[1]);

            updateGateInput[i][t].addMulRowToMatrix(input[i], t, weightInput[0]);
            resetGateInput[i][t].addMulRowToMatrix(input[i], t, weightInput[1]);
            if (hidden_t != null) {
                updateGateInput[i][t].addMul(hidden_t, weightHidden[0]);
                resetGateInput[i][t].addMul(hidden_t, weightHidden[1]);
            }

            //activation update and reset gate
            functionActivationSigmoid.activation(updateGateInput[i][t], updateGateOutput[i][t]);
            functionActivationSigmoid.activation(resetGateInput[i][t], resetGateOutput[i][t]);

            inputHidden[i][t].set(threshold[2]);
            inputHidden[i][t].addMulRowToMatrix(input[i], t, weightInput[2]);
            if (hidden_t != null) {
                resetHidden[i][t].mulVectors(hidden_t, resetGateOutput[i][t]);
                inputHidden[i][t].addMul(resetHidden[i][t], weightHidden[2]);
            }

            //find output memory content
            functionActivationTanh.activation(inputHidden[i][t], outputHidden[i][t]);

            //find current hidden state
            hidden[i][t].setMulUpdateVectors(updateGateOutput[i][t], outputHidden[i][t]);
            if (hidden_t != null) {
                hidden[i][t].addProduct(updateGateOutput[i][t], hidden_t);
            }

            //dropout hidden state
            if (dropout && recurrentDropout != 0) {
                hidden[i][t].dropout(hidden[i][t], recurrentDropout);
            }
            outputVector[i][t] = generateAttention(t, i, hidden[i][t]).concat(hidden[i][t]);

            //if return sequence pass current hidden state to output
            if (returnSequences || t == input[i].getRow() - 1) {
                output[i].set(outputVector[i][t], tOut);
                tOut++;
            }
        }
        //if layer return state,than save last hidden state
        state[i][0] = outputHidden[i][input[i].getRow() - 1];
    }

    @Override
    public void generateError(NNArray[] errors, NNArray[][] errorsState) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[input.length];
        this.errorState = new NNVector[input.length][1];

        ExecutorService executor = Executors.newFixedThreadPool(errors.length);
        for (int cor = 0; cor < errors.length; cor++) {
            final int i = cor;
            executor.execute(() -> {
                if (errorsState != null) {
                    generateError(i, NNArrays.isVector(errorsState[i]));
                } else {
                    generateError(i, null);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        //regularization derivative weightAttention
        if (trainable && regularization != null) {
            for (int i = 0; i < 3; i++) {
                regularization.regularization(weightInput[i]);
                regularization.regularization(weightHidden[i]);
                regularization.regularization(threshold[i]);
            }
            if (attention == Attention.GENERAL) {
                regularization.regularization(weightVal);
            } else if (attention == Attention.CONCAT) {
                regularization.regularization(weightAttention);
                regularization.regularization(weightVal);
            }
        }
    }

    private void generateError(int i, NNVector[] errorState) {
        this.error[i] = new NNMatrix(input[i]);
        this.errorInput[i] = new NNMatrix(outputPreLayer[i]);
        NNVector hiddenError = new NNVector(countNeuron);
        NNVector resetDelta = new NNVector(countNeuron);
        NNVector resetError = new NNVector(countNeuron);
        NNVector hiddenHDelta = new NNVector(countNeuron);
        NNVector hiddenHError = new NNVector(countNeuron);
        NNVector updateDelta = new NNVector(countNeuron);
        NNVector updateError = new NNVector(countNeuron);
        NNVector resetHiddenError = new NNVector(countNeuron);

        //copy error from next layer
        int tError = (returnSequences) ? hidden[i].length - 1 : 0;
        NNVector[] hiddenErrors = errorNL[i].toVectors();
        hiddenError = hiddenErrors[tError].subVector(countNeuron, countNeuron);
        if (errorState != null) {
            hiddenError.add(errorState[0]);
        }
        hiddenError.add(generateErrorAttention(hiddenErrors[tError].subVector(0, countNeuron), i, tError));

        //pass through time
        for (int t = hidden[i].length - 1; t >= 0; t--) {
            NNVector hidden_t = null;
            if (t > 0) {
                hidden_t = hidden[i][t - 1];
            } else if (inputState[i] != null) {
                hidden_t = inputState[i][0];
            }
            //dropout back for error
            if(recurrentDropout != 0) {
                hiddenError.dropoutBack(hidden[i][t], hiddenError, recurrentDropout);
            }
            //find error for update and reset gate
            updateError.mulNegativeVectors(hiddenError, outputHidden[i][t]);
            if (hidden_t != null) {
                updateError.addProduct(hiddenError, hidden_t);
            }
            hiddenHError.setMulUpdateVectors(updateGateOutput[i][t], hiddenError);

            //derivative activation for current time step
            functionActivationSigmoid.derivativeActivation(updateGateInput[i][t], updateGateOutput[i][t], updateError, updateDelta);
            functionActivationTanh.derivativeActivation(inputHidden[i][t], outputHidden[i][t], hiddenHError, hiddenHDelta);

            //find error for reset gate
            resetDelta.clear();
            resetHiddenError.clear();
            resetHiddenError.addMulT(hiddenHDelta, weightHidden[2]);
            if (hidden_t != null) {
                resetError.mulVectors(resetHiddenError, hidden_t);
                functionActivationSigmoid.derivativeActivation(resetGateInput[i][t], resetGateOutput[i][t], resetError, resetDelta);
            }

            //find derivative for weightAttention
            if (trainable) {
                derivativeWeight(t, i, hidden_t, updateDelta, resetDelta, hiddenHDelta);
            }

            //find error for previous time step
            hiddenError.mul(updateGateOutput[i][t]);
            if (returnSequences && t > 0 && errorNL != null) {
                hiddenError.add(hiddenErrors[t - 1].subVector(countNeuron, countNeuron));
                hiddenError.add(generateErrorAttention(hiddenErrors[t - 1].subVector(0, countNeuron), i, t - 1));
            }
            hiddenError.addProduct(resetHiddenError, resetGateOutput[i][t]);
            hiddenError.addMulT(updateDelta, weightHidden[0]);
            hiddenError.addMulT(resetDelta, weightHidden[1]);

            //find error for previous layer
            error[i].addMulT(t, updateDelta, weightInput[0]);
            error[i].addMulT(t, resetDelta, weightInput[1]);
            error[i].addMulT(t, hiddenHDelta, weightInput[2]);
        }
        this.errorState[i][0] = hiddenError;
    }

    private void derivativeWeight(int t, int i, NNVector hidden_t, NNVector updateDelta, NNVector resetDelta, NNVector hiddenHDelta) {
        derThreshold[0].add(updateDelta);
        derThreshold[1].add(resetDelta);
        derThreshold[2].add(hiddenHDelta);
        int indexHWeight = 0, indexIWeight = 0, indexInput;

        for (int k = 0; k < hidden[i][t].size(); k++) {
            indexInput = input[i].getRowIndex()[t];

            if (hidden_t != null) {
                //find derivative for hidden weightAttention
                for (int m = 0; m < countNeuron; m++, indexHWeight++) {
                    derWeightHidden[0].getData()[indexHWeight] += updateDelta.get(k) * hidden_t.get(m);
                    derWeightHidden[1].getData()[indexHWeight] += resetDelta.get(k) * hidden_t.get(m);
                    derWeightHidden[2].getData()[indexHWeight] += hiddenHDelta.get(k) * resetHidden[i][t].get(m);
                }
            }
            //find derivative for input's weightAttention
            for (int m = 0; m < input[i].getColumn(); m++, indexIWeight++, indexInput++) {
                derWeightInput[0].getData()[indexIWeight] += updateDelta.get(k) * input[i].getData()[indexInput];
                derWeightInput[1].getData()[indexIWeight] += resetDelta.get(k) * input[i].getData()[indexInput];
                derWeightInput[2].getData()[indexIWeight] += hiddenHDelta.get(k) * input[i].getData()[indexInput];
            }
        }
    }

    public GRULuongAttentionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    public GRULuongAttentionLayer setInitializer(Initializer initializer) {
        this.initializerInput = initializer;
        this.initializerHidden = initializer;

        return this;
    }

    public GRULuongAttentionLayer setInitializerInput(Initializer initializer) {
        this.initializerInput = initializer;

        return this;
    }

    public GRULuongAttentionLayer setInitializerHidden(Initializer initializer) {
        this.initializerHidden = initializer;

        return this;
    }

    public GRULuongAttentionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public static GRULuongAttentionLayer read(Scanner scanner) {
        GRULuongAttentionLayer recurrentLayer = new GRULuongAttentionLayer(Integer.parseInt(scanner.nextLine()),
                Double.parseDouble(scanner.nextLine()),
                readAttention(scanner.nextLine()),
                Boolean.parseBoolean(scanner.nextLine()));

        recurrentLayer.threshold = new NNVector[3];
        recurrentLayer.weightInput = new NNMatrix[3];
        recurrentLayer.weightHidden = new NNMatrix[3];

        for (int i = 0; i < 3; i++) {
            recurrentLayer.threshold[i] = NNVector.read(scanner);
            recurrentLayer.weightInput[i] = NNMatrix.read(scanner);
            recurrentLayer.weightHidden[i] = NNMatrix.read(scanner);
        }

        if (recurrentLayer.attention == Attention.GENERAL) {
            recurrentLayer.weightVal = NNMatrix.read(scanner);
        } else if (recurrentLayer.attention == Attention.CONCAT) {
            recurrentLayer.weightVal = NNMatrix.read(scanner);
            recurrentLayer.weightAttention = NNMatrix.read(scanner);
        }

        recurrentLayer.setRegularization(Regularization.read(scanner));
        recurrentLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        recurrentLayer.loadWeight = true;
        return recurrentLayer;
    }
}