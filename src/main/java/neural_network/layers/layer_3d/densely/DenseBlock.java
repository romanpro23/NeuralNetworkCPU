package neural_network.layers.layer_3d.densely;

import lombok.Getter;
import neural_network.layers.layer_3d.NeuralLayer3D;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class DenseBlock extends NeuralLayer3D {
    @Getter
    private ArrayList<DenseUnit> module;

    public DenseBlock() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (DenseUnit denseUnit : module) {
            denseUnit.initialize(optimizer);
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

        for (DenseUnit denseUnit : module) {
            denseUnit.initialize(new int[]{outHeight, outWidth, outDepth});
            outDepth += denseUnit.size()[2];
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);

        module.get(0).generateOutput(input);
        this.output = NNArrays.concatTensor(this.input, module.get(0).getOutput());
        for (int i = 1; i < module.size(); i++) {
            module.get(i).generateOutput(output);
            this.output = NNArrays.concatTensor(this.output, module.get(i).getOutput());
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);

        module.get(0).generateTrainOutput(this.input);
        this.output = NNArrays.concatTensor(this.input, module.get(0).getOutput());
        for (int i = 1; i < module.size(); i++) {
            module.get(i).generateTrainOutput(output);
            this.output = NNArrays.concatTensor(this.output, module.get(i).getOutput());
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);

        int index = output[0].size();
        for (int i = module.size() - 1; i >= 0; i--) {
            index -= module.get(i).getOutput()[0].size();
            module.get(i).generateError(NNArrays.subTensor(errorNL, module.get(i).getOutput(), index));
            addErrorBlock(NNArrays.isTensor(module.get(i).getError()));
        }

        this.error = NNArrays.subTensor(errorNL, input, 0);
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    private void addErrorBlock(NNTensor[] errorBlock) {
        int row = errorNL[0].getRows();
        int column = errorNL[0].getColumns();
        int depth = errorBlock[0].shape()[2];
        int index, indexSubTensor = 0;
        for (int i = 0; i < errorNL.length; i++) {
            indexSubTensor = 0;
            for (int j = 0; j < row; j++) {
                for (int k = 0; k < column; k++) {
                    index = errorNL[i].getRowsIndex()[j] + errorNL[i].getColumnsIndex()[k];
                    for (int d = 0; d < depth; d++, index++, indexSubTensor++) {
                        errorNL[i].getData()[index] += errorBlock[i].getData()[indexSubTensor];
                    }
                }
            }
        }
    }

    @Override
    public void trainable(boolean trainable) {
        for (DenseUnit denseUnit : module) {
            denseUnit.trainable(trainable);
        }
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Dense block\n");
        for (DenseUnit layer : module) {
            layer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |           Dense block         |             ");
        System.out.println("____________|_______________________________|_____________");
        for (DenseUnit denseUnit : module) {
            countParam += denseUnit.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public DenseBlock addDenseUnit(DenseUnit denseUnit) {
        module.add(denseUnit);
        return this;
    }

    public DenseBlock setTrainable(boolean trainable) {
        for (DenseUnit denseUnit : module) {
            denseUnit.setTrainable(trainable);
        }
        return this;
    }

    public static DenseBlock read(Scanner scanner) {
        DenseBlock denseBlock = new DenseBlock();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            denseBlock.module.add(DenseUnit.read(scanner));
            layer = scanner.nextLine();
        }

        return denseBlock;
    }
}
