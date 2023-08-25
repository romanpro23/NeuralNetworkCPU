package neural_network.layers.layer_3d.residual;

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

public class ResidualBlock extends NeuralLayer3D {
    @Getter
    private ArrayList<ResidualUnit> module;

    public ResidualBlock() {
        module = new ArrayList<>();
    }

    @Override
    public void initialize(Optimizer optimizer) {
        for (ResidualUnit residualUnit : module) {
            residualUnit.initialize(optimizer);
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

        for (ResidualUnit residualUnit : module) {
            residualUnit.initialize(size);
        }

        outHeight = module.get(0).size()[0];
        outWidth = module.get(0).size()[1];
        outDepth = module.get(0).size()[2];
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isTensor(inputs);
        for (ResidualUnit residualUnit : module) {
            residualUnit.generateOutput(inputs);
        }
        generateOutputModule();
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    private void generateOutputModule() {
        if (module.size() == 1) {
            this.output = NNArrays.isTensor(module.get(0).getOutput());
        } else {
            this.output = new NNTensor[input.length];
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNTensor(outHeight, outWidth, outDepth);
                for (ResidualUnit residualUnit : module) {
                    output[i].add(residualUnit.getOutput()[i]);
                }
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);
        for (ResidualUnit residualUnit : module) {
            residualUnit.generateTrainOutput(input);
        }
        generateOutputModule();
    }

    @Override
    public void generateError(NNArray[] errors) {
        this.errorNL = getErrorNextLayer(errors);
        this.error = new NNTensor[errorNL.length];
        for (ResidualUnit residualUnit : module) {
            residualUnit.generateError(errorNL);
        }

        for (int i = 0; i < errorNL.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            for (ResidualUnit residualUnit : module) {
                error[i].add(residualUnit.getError()[i]);
            }
        }
    }

    @Override
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Residual block\n");
        for (ResidualUnit layer : module) {
            layer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |          Residual block       |             ");
        System.out.println("____________|_______________________________|_____________");
        for (ResidualUnit residualUnit : module) {
            countParam += residualUnit.info();
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        return countParam;
    }

    public ResidualBlock addResidualUnit(ResidualUnit residualUnit) {
        module.add(residualUnit);
        return this;
    }

    public ResidualBlock setTrainable(boolean trainable) {
        for (ResidualUnit residualUnit : module) {
            residualUnit.setTrainable(trainable);
        }
        return this;
    }    

    @Override
    public void trainable(boolean trainable) {
        for (ResidualUnit residualUnit : module) {
            residualUnit.trainable(trainable);
        }
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    public static ResidualBlock read(Scanner scanner) {
        ResidualBlock residualBlock = new ResidualBlock();

        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            residualBlock.module.add(ResidualUnit.read(scanner));
            layer = scanner.nextLine();
        }

        return residualBlock;
    }
}
