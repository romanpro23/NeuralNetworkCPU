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
        }
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

    private void generateOutput(int i, NNVector[] states) {
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
            //if return sequence pass current hidden state to output
            if (returnSequences || t == input[i].getRow() - 1) {
                output[i].set(hidden[i][t], tOut);
                tOut++;
            }
        }
        //if layer return state,than save last hidden state
        state[i][0] = outputHidden[i][input[i].getRow() - 1];
    }

    private void generateError(int i, NNVector[] errorState) {
        this.error[i] = new NNMatrix(input[i]);
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
        if (errorNL != null) {
            hiddenError.setRowFromMatrix(errorNL[i], tError);
        }
        if (errorState != null) {
            hiddenError.add(errorState[0]);
        }

        //pass through time
        for (int t = hidden[i].length - 1; t >= 0; t--) {
            NNVector hidden_t = null;
            if (t > 0) {
                hidden_t = hidden[i][t - 1];
            } else if (inputState[i] != null) {
                hidden_t = inputState[i][0];
            }
            //dropout back for error
            if (recurrentDropout != 0) {
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
                hiddenError.addRowFromMatrix(errorNL[i], t - 1);
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