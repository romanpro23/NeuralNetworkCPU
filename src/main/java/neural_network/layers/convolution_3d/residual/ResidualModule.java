package neural_network.layers.convolution_3d.residual;

import lombok.Getter;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.layers.convolution_3d.inception.InceptionBlock;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class ResidualModule extends ConvolutionNeuralLayer {
    @Getter
    private ArrayList<ResidualBlock> module;

    public ResidualModule() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (ResidualBlock residualBlock : module) {
            residualBlock.initialize(optimizer);
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
        outHeight = height;
        outWidth = width;

        for (ResidualBlock residualBlock : module) {
            residualBlock.initialize(size);
        }
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        for (ResidualBlock residualBlock : module) {
            residualBlock.generateOutput(inputs);
        }
        generateOutputModule();
    }

    private void generateOutputModule() {
        if (module.size() == 1) {
            this.output = NNArrays.isTensor(module.get(0).getOutput());
        } else {
            this.output = new NNTensor[input.length];
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNTensor(height, width, depth);
                for (ResidualBlock residualBlock : module) {
                    output[i].add(residualBlock.getOutput()[i]);
                }
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        for (ResidualBlock residualBlock : module) {
            residualBlock.generateTrainOutput(input);
        }
        generateOutputModule();
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errorNL.length];
        int index = 0;
        for (ResidualBlock residualBlock : module) {
            residualBlock.generateError(errorNL);
        }

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            for (ResidualBlock residualBlock : module) {
                error[i].add(residualBlock.getError()[i]);
            }
        }
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Residual module\n");
        for (ResidualBlock layer : module) {
            layer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |          Residual module      |             ");
        System.out.println("____________|_______________________________|_____________");
        for (ResidualBlock residualBlock : module) {
            countParam += residualBlock.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public ResidualModule addResidualBlock(ResidualBlock residualBlock) {
        module.add(residualBlock);
        return this;
    }

    public ResidualModule setTrainable(boolean trainable) {
        for (ResidualBlock residualBlock : module) {
            residualBlock.setTrainable(trainable);
        }
        return this;
    }

    public static ResidualModule read(Scanner scanner) {
        ResidualModule residualModule = new ResidualModule();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            residualModule.module.add(ResidualBlock.read(scanner));
            layer = scanner.nextLine();
        }

        return residualModule;
    }
}
