package data.svhn;

import data.loaders.*;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class SVHNLoader3D extends DataLoader3D {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[3072];
    private byte[] byteNumb = new byte[1];

    private static FileInputStream scanner;

    public SVHNLoader3D() {
        this(new TransformData.Sigmoid());
    }

    public SVHNLoader3D(TransformData transform) {
        this.transformData = transform;

        train = new ArrayList<>(73257);
        test = new ArrayList<>(26032);

        try {
            loadTrainData();
            loadTestData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
        scanner = new FileInputStream("D:/datasets/svhn/train.bin");
        for (int i = 0; i < 73257; i++) {
            if (scanner.available() >= 3072) {
                scanner.read(bytes);
                scanner.read(byteNumb);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(10);
                output.set(trueNumb, 1);

                NNTensor input = new NNTensor(32, 32, 3);
                int index = 0;
                for (int j = 0; j < 3; j++) {
                    for (int l = 0; l < 32; l++) {
                        for (int k = 0; k < 32; k++, index++) {
                            input.set(l, k, j, inputsData[index]);
                        }
                    }
                }

                train.add(new ImageData3D(input, output));
            }
        }
        Collections.shuffle(train);
    }

    private void loadTestData() throws IOException {
        scanner = new FileInputStream("D:/datasets/svhn/test.bin");

        for (int i = 0; i < 26032; i++) {
            if (scanner.available() >= 3072) {
                scanner.read(bytes);
                scanner.read(byteNumb);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(10);
                output.set(trueNumb, 1);

                NNTensor input = new NNTensor(32, 32, 3);
                int index = 0;
                for (int j = 0; j < 3; j++) {
                    for (int l = 0; l < 32; l++) {
                        for (int k = 0; k < 32; k++, index++) {
                            input.set(l, k, j, inputsData[index]);
                        }
                    }
                }

                test.add(new ImageData3D(input, output));
            }
        }
        Collections.shuffle(test);
    }

    private void generateInput() {
        inputsData = new float[3072];
        for (int i = 0; i < 1024; i++) {
            inputsData[i] = transformData.transformR(bytes[i]);
        }
        for (int i = 1024; i < 2048; i++) {
            inputsData[i] = transformData.transformG(bytes[i]);
        }
        for (int i = 2048; i < 3072; i++) {
            inputsData[i] = transformData.transformB(bytes[i]);
        }
    }
}