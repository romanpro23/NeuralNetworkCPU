package neural_network.layers.layer_3d.densely;

import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.NeuralLayer3D;
import neural_network.optimizers.Optimizer;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ResidualDenseBlock extends NeuralLayer3D {
    private DenseBlock denseBlock;

    private ConvolutionLayer outputLayer;

    private NNVector betta;
    private NNVector derBetta;
    private boolean trainableBetta;

    private boolean loadWeight;

    public ResidualDenseBlock() {
        denseBlock = new DenseBlock();

        trainableBetta = false;
    }

    @Override
    public void initialize(Optimizer optimizer) {
        denseBlock.initialize(optimizer);
        outputLayer.initialize(optimizer);
        if (trainableBetta) {
            optimizer.addDataOptimize(betta, derBetta);
        }
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];

        outDepth = depth;
        outWidth = width;
        outHeight = height;

        denseBlock.initialize(size);
        outputLayer = new ConvolutionLayer(depth, 3, 1, 1);
        outputLayer.initialize(denseBlock.size());

        derBetta = new NNVector(1);
        if(!loadWeight) {
            betta = new NNVector(1);
            betta.fill(0.2f);
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        this.output = new NNTensor[inputs.length];

        denseBlock.generateOutput(inputs);
        outputLayer.generateOutput(denseBlock.getOutput());

        generateOutputBlock();
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        this.output = new NNTensor[inputs.length];

        denseBlock.generateTrainOutput(inputs);
        outputLayer.generateTrainOutput(denseBlock.getOutput());

        generateOutputBlock();
    }

    private void generateOutputBlock() {
        for (int i = 0; i < input.length; i++) {
            output[i] = new NNTensor(height, width, depth);
            output[i].add(input[i]);
            output[i].addMul(outputLayer.getOutput()[i], betta.get(0));
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);
        generateErrorBlock(errors);

        outputLayer.generateError(errorNL);
        denseBlock.generateError(outputLayer.getError());

        addError();
    }

    private void addError(){
        for (int i = 0; i < error.length; i++) {
            this.error[i].add(denseBlock.getError()[i]);
        }
    }

    private void generateErrorBlock(NNArray[] errors){
        this.error = new NNTensor[errors.length];
        for (int i = 0; i < errors.length; i++) {
            this.error[i] = new NNTensor(height, width, depth);
            this.error[i].add(errorNL[i]);
            if(trainableBetta){
                derivativeBetta(outputLayer.getOutput()[i], errorNL[i]);
            }
            errorNL[i].mul(betta.get(0));
        }
    }

    private void derivativeBetta(NNArray out, NNArray error) {
        for (int i = 0; i < out.size(); i++) {
            derBetta.getData()[0] += out.get(i) * error.get(i);
        }
    }

    @Override
    public void trainable(boolean trainable) {
        denseBlock.trainable(trainable);
        outputLayer.setTrainable(trainable);
        if (!trainable) {
            trainableBetta = false;
        }
    }

    public ResidualDenseBlock setTrainableBetta(boolean trainable) {
        this.trainableBetta = trainable;

        return this;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Residual dense block\n");
        writer.write(trainableBetta + "\n");
        betta.save(writer);
        denseBlock.save(writer);
        outputLayer.save(writer);
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |      Residual dense block     |             ");
        System.out.println("____________|_______________________________|_____________");
        countParam += denseBlock.info();
        System.out.println("____________|_______________________________|_____________");
        countParam += outputLayer.info();
        System.out.println("____________|_______________________________|_____________");
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public ResidualDenseBlock addDenseUnit(DenseUnit denseUnit) {
        denseBlock.addDenseUnit(denseUnit);
        return this;
    }

    public ResidualDenseBlock setTrainable(boolean trainable) {
        this.trainable(trainable);
        return this;
    }

    public static ResidualDenseBlock read(Scanner scanner) {
        ResidualDenseBlock denseBlock = new ResidualDenseBlock();

        denseBlock.loadWeight = false;
        denseBlock.trainableBetta = Boolean.parseBoolean(scanner.nextLine());
        denseBlock.betta = NNVector.read(scanner);
        denseBlock.denseBlock = DenseBlock.read(scanner);
        denseBlock.outputLayer = ConvolutionLayer.read(scanner);
        denseBlock.trainable(Boolean.parseBoolean(scanner.nextLine()));
        denseBlock.loadWeight = true;

        return denseBlock;
    }
}
