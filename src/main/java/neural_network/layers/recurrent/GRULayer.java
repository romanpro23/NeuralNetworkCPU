package neural_network.layers.recurrent;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GRULayer extends RecurrentNeuralLayer {
    private NNVector[][] hidden;
    private NNVector[][] resetHidden;
    private NNVector[][] updateGateInput;
    private NNVector[][] updateGateOutput;
    private NNVector[][] resetGateInput;
    private NNVector[][] resetGateOutput;

    private NNVector[] resetDelta;
    private NNVector[] resetError;
    private NNVector[] hiddenHDelta;
    private NNVector[] hiddenHError;
    private NNVector[] updateDelta;
    private NNVector[] updateError;
    private NNVector[] resetHiddenError;

    private NNMatrix[] weightInput;
    private NNMatrix[] derWeightInput;

    private NNMatrix[] weightHidden;
    private NNMatrix[] derWeightHidden;

    private NNVector[] threshold;
    private NNVector[] derThreshold;

    private final FunctionActivation functionActivationSigmoid;
    private final FunctionActivation functionActivationTanh;

    public GRULayer(int countNeuron) {
        this(countNeuron, 0);
    }

    public GRULayer(GRULayer layer) {
        this(layer.countNeuron, layer.recurrentDropout, layer.returnSequences);
        this.copy(layer);
    }

    public GRULayer(int countNeuron, double recurrentDropout) {
        super(countNeuron, recurrentDropout);

        this.functionActivationTanh = new FunctionActivation.Tanh();
        this.functionActivationSigmoid = new FunctionActivation.Sigmoid();
    }

    public GRULayer(int countNeuron, double recurrentDropout, boolean returnSequences) {
        this(countNeuron, recurrentDropout);
        setReturnSequences(returnSequences);
    }

    public GRULayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;

        return this;
    }

    public GRULayer setPreLayer(RecurrentNeuralLayer layer) {
        super.setPreLayer(layer);

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
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (int i = 0; i < 3; i++) {
            optimizer.addDataOptimize(weightInput[i], derWeightInput[i]);
            optimizer.addDataOptimize(weightHidden[i], derWeightHidden[i]);
            optimizer.addDataOptimize(threshold[i], derThreshold[i]);
        }
    }

    @Override
    public int info() {
        int countParam = (weightHidden[0].size() + weightInput[0].size() + threshold[0].size()) * 3;
        System.out.println("GRU\t\t\t|  " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + countNeuron + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("GRU layer\n");
        writer.write(countNeuron + "\n");
        writer.write(recurrentDropout + "\n");
        writer.write(returnSequences + "\n");
        writer.write(returnState + "\n");

        for (int i = 0; i < 3; i++) {
            threshold[i].save(writer);
            weightInput[i].save(writer);
            weightHidden[i].save(writer);
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
    public void generateOutput(NNArray[] inputs) {
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
        if (returnState) {
            this.state = new NNVector[inputs.length][1];
        }

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int cor = 0; cor < countC; cor++) {
            final int firstIndex = cor * input.length / countC;
            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    generateOutput(i);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void generateOutput(int i) {
        int countRow = (returnSequences) ? input[i].getRow() : 1;
        output[i] = new NNMatrix(countRow, countNeuron);

        inputHidden[i] = new NNVector[input[i].getRow()];
        outputHidden[i] = new NNVector[input[i].getRow()];

        this.hidden[i] = new NNVector[input[i].getRow()];
        this.resetHidden[i] = new NNVector[input[i].getRow()];
        this.resetGateInput[i] = new NNVector[input[i].getRow()];
        this.resetGateOutput[i] = new NNVector[input[i].getRow()];
        this.updateGateInput[i] = new NNVector[input[i].getRow()];
        this.updateGateOutput[i] = new NNVector[input[i].getRow()];

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
            } else if (hasPreLayer()) {
                hidden_t = getStatePreLayer(i)[0];
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
            if (dropout) {
                hidden[i][t].dropout(hidden[i][t], recurrentDropout);
            }
            //if return sequence pass current hidden state to output
            if (returnSequences || t == input[i].getRow() - 1) {
                output[i].set(hidden[i][t], tOut);
                tOut++;
            }
        }
        //if layer return state,than save last hidden state
        if (returnState) {
            state[i][0] = hidden[i][input[i].getRow() - 1];
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[input.length];
        this.hiddenError = new NNVector[input.length];
        if (hasPreLayer()) {
            this.errorState = new NNVector[input.length][1];
        }

        resetDelta = new NNVector[input.length];
        resetError = new NNVector[input.length];
        hiddenHDelta = new NNVector[input.length];
        hiddenHError = new NNVector[input.length];
        updateDelta = new NNVector[input.length];
        updateError = new NNVector[input.length];
        resetHiddenError = new NNVector[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int cor = 0; cor < countC; cor++) {
            final int firstIndex = cor * input.length / countC;
            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    generateError(i);
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
        }
    }

    private void generateError(int i) {
        this.error[i] = new NNMatrix(input[i]);
        hiddenError[i] = new NNVector(countNeuron);
        resetDelta[i] = new NNVector(countNeuron);
        resetError[i] = new NNVector(countNeuron);
        hiddenHDelta[i] = new NNVector(countNeuron);
        hiddenHError[i] = new NNVector(countNeuron);
        updateDelta[i] = new NNVector(countNeuron);
        updateError[i] = new NNVector(countNeuron);
        resetHiddenError[i] = new NNVector(countNeuron);

        //copy error from next layer
        int tError = (returnSequences) ? hidden[i].length - 1 : 0;
        if (errorNL != null) {
            hiddenError[i].setRowFromMatrix(errorNL[i], tError);
        }
        if (returnState) {
            hiddenError[i].add(getErrorStateNextLayer(i)[0]);
        }

        //pass through time
        for (int t = hidden[i].length - 1; t >= 0; t--) {
            NNVector hidden_t = null;
            if (t > 0) {
                hidden_t = hidden[i][t - 1];
            } else if (hasPreLayer()) {
                hidden_t = getStatePreLayer(i)[0];
            }
            //dropout back for error
            hiddenError[i].dropoutBack(hidden[i][t], hiddenError[i], recurrentDropout);
            //find error for update and reset gate
            updateError[i].mulNegativeVectors(hiddenError[i], outputHidden[i][t]);
            if (hidden_t != null) {
                updateError[i].addProduct(hiddenError[i], hidden_t);
            }
            hiddenHError[i].setMulUpdateVectors(updateGateOutput[i][t], hiddenError[i]);

            //derivative activation for current time step
            functionActivationSigmoid.derivativeActivation(updateGateInput[i][t], updateGateOutput[i][t], updateError[i], updateDelta[i]);
            functionActivationTanh.derivativeActivation(inputHidden[i][t], outputHidden[i][t], hiddenHError[i], hiddenHDelta[i]);

            //find error for reset gate
            resetDelta[i].clear();
            resetHiddenError[i].clear();
            resetHiddenError[i].addMulT(hiddenHDelta[i], weightHidden[2]);
            if (hidden_t != null) {
                resetError[i].mulVectors(resetHiddenError[i], hidden_t);
                functionActivationSigmoid.derivativeActivation(resetGateInput[i][t], resetGateOutput[i][t], resetError[i], resetDelta[i]);
            }

            //find derivative for weightAttention
            if (trainable) {
                derivativeWeight(t, i, hidden_t);
            }

            //find error for previous time step
            hiddenError[i].mul(updateGateOutput[i][t]);
            if (returnSequences && t > 0 && errorNL != null) {
                hiddenError[i].addRowFromMatrix(errorNL[i], t - 1);
            }
            hiddenError[i].addProduct(resetHiddenError[i], resetGateOutput[i][t]);
            hiddenError[i].addMulT(updateDelta[i], weightHidden[0]);
            hiddenError[i].addMulT(resetDelta[i], weightHidden[1]);

            //find error for previous layer
            error[i].addMulT(t, updateDelta[i], weightInput[0]);
            error[i].addMulT(t, resetDelta[i], weightInput[1]);
            error[i].addMulT(t, hiddenHDelta[i], weightInput[2]);

            if (t == 0 && hasPreLayer()) {
                errorState[i][0] = new NNVector(countNeuron);
                errorState[i][0].set(this.hiddenError[i]);
            }
        }
    }

    private void derivativeWeight(int t, int i, NNVector hidden_t) {
        derThreshold[0].add(updateDelta[i]);
        derThreshold[1].add(resetDelta[i]);
        derThreshold[2].add(hiddenHDelta[i]);
        int indexHWeight = 0, indexIWeight = 0, indexInput;

        for (int k = 0; k < hidden[i][t].size(); k++) {
            indexInput = input[i].getRowIndex()[t];

            if (hidden_t != null) {
                //find derivative for hidden weightAttention
                for (int m = 0; m < countNeuron; m++, indexHWeight++) {
                    derWeightHidden[0].getData()[indexHWeight] += updateDelta[i].get(k) * hidden_t.get(m);
                    derWeightHidden[1].getData()[indexHWeight] += resetDelta[i].get(k) * hidden_t.get(m);
                    derWeightHidden[2].getData()[indexHWeight] += hiddenHDelta[i].get(k) * resetHidden[i][t].get(m);
                }
            }
            //find derivative for input's weightAttention
            for (int m = 0; m < input[i].getColumn(); m++, indexIWeight++, indexInput++) {
                derWeightInput[0].getData()[indexIWeight] += updateDelta[i].get(k) * input[i].getData()[indexInput];
                derWeightInput[1].getData()[indexIWeight] += resetDelta[i].get(k) * input[i].getData()[indexInput];
                derWeightInput[2].getData()[indexIWeight] += hiddenHDelta[i].get(k) * input[i].getData()[indexInput];
            }
        }
    }

    public GRULayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    public GRULayer setInitializer(Initializer initializer) {
        this.initializerInput = initializer;
        this.initializerHidden = initializer;

        return this;
    }

    public GRULayer setInitializerInput(Initializer initializer) {
        this.initializerInput = initializer;

        return this;
    }

    public GRULayer setInitializerHidden(Initializer initializer) {
        this.initializerHidden = initializer;

        return this;
    }

    public GRULayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public static GRULayer read(Scanner scanner) {
        GRULayer recurrentLayer = new GRULayer(Integer.parseInt(scanner.nextLine()),
                Double.parseDouble(scanner.nextLine()),
                Boolean.parseBoolean(scanner.nextLine()));

        recurrentLayer.returnState = Boolean.parseBoolean(scanner.nextLine());

        recurrentLayer.threshold = new NNVector[3];
        recurrentLayer.weightInput = new NNMatrix[3];
        recurrentLayer.weightHidden = new NNMatrix[3];

        for (int i = 0; i < 3; i++) {
            recurrentLayer.threshold[i] = NNVector.read(scanner);
            recurrentLayer.weightInput[i] = NNMatrix.read(scanner);
            recurrentLayer.weightHidden[i] = NNMatrix.read(scanner);
        }

        recurrentLayer.setRegularization(Regularization.read(scanner));
        recurrentLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        recurrentLayer.loadWeight = true;
        return recurrentLayer;
    }
}