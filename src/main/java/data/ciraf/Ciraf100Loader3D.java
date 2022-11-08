package data.ciraf;

import data.loaders.*;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Ciraf100Loader3D extends DataLoader3D {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[3072];
    private byte[] byteNumb = new byte[2];

    private static FileInputStream scanner;
    private static FileInputStream scannerTest;

    private void loadTrainFilesWithNumber() {
        try {
            scanner = new FileInputStream("D:/datasets/ciraf/train.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTestFilesWithNumber() {
        try {
            scannerTest = new FileInputStream("D:/datasets/ciraf/test.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Ciraf100Loader3D() {
        this(new TransformData.Sigmoid());
    }

    public Ciraf100Loader3D(TransformData transform) {
        this.transformData = transform;

        train = new ArrayList<>(50000);
        test = new ArrayList<>(10000);

        try {
            loadTrainData();
            loadTestData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
       loadTrainFilesWithNumber();
        for (int i = 0; i < 50000; i++) {
            if (scanner.available() >= 3072) {
                scanner.read(byteNumb);
                scanner.read(bytes);

                trueNumb = byteNumb[1];
                generateInput();
                NNVector output = new NNVector(100);
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
        loadTestFilesWithNumber();
        for (int i = 0; i < 10000; i++) {
            if (scannerTest.available() >= 3072) {
                scannerTest.read(byteNumb);
                scannerTest.read(bytes);
                trueNumb = byteNumb[1];
                generateInput();
                NNVector output = new NNVector(100);
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
        for (int i = 0; i < 3072; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
    }
}