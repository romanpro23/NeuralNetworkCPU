package neural_network.layers.convolution_3d.densely;

import lombok.Getter;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class DenseModule extends ConvolutionNeuralLayer {
    @Getter
    private ArrayList<DenseBlock> module;

    public DenseModule() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (DenseBlock denseBlock : module) {
            denseBlock.initialize(optimizer);
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
        for (DenseBlock denseBlock : module) {
            denseBlock.initialize(new int[]{outHeight, outWidth, outDepth});
            outDepth += denseBlock.size()[2];
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

    private void addErrorBlock(NNTensor[] errorBlock) {
        int row = errorNL[0].getRows();
        int column = errorNL[0].getColumns();
        int depth = errorBlock[0].getSize()[2];
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
    public void write(FileWriter writer) throws IOException {
        writer.write("Dense module\n");
        for (DenseBlock layer : module) {
            layer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |           Dense module        |             ");
        System.out.println("____________|_______________________________|_____________");
        for (DenseBlock denseBlock : module) {
            countParam += denseBlock.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public DenseModule addDenseBlock(DenseBlock denseBlock) {
        module.add(denseBlock);
        return this;
    }

    public DenseModule setTrainable(boolean trainable) {
        for (DenseBlock denseBlock : module) {
            denseBlock.setTrainable(trainable);
        }
        return this;
    }

    public static DenseModule read(Scanner scanner) {
        DenseModule denseModule = new DenseModule();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            denseModule.module.add(DenseBlock.read(scanner));
            layer = scanner.nextLine();
        }

        return denseModule;
    }
}
