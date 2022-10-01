package data.svhn;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.loaders.TransformData;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class SVHNLoader1D extends DataLoader1D {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[3072];
    private byte[] byteNumb = new byte[1];

    private static FileInputStream scanner;

    public SVHNLoader1D() {
        this(new TransformData.Sigmoid());
    }

    public SVHNLoader1D(TransformData transform) {
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

                train.add(new ImageData1D(new NNVector(inputsData), output));
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

                test.add(new ImageData1D(new NNVector(inputsData), output));
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