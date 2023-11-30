package data.quick_draw;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.loaders.TransformData;
import lombok.Setter;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class QuickDrawLoader1D extends DataLoader1D {
    @Setter
    private TransformData transformData;
    private float[] inputsData;
    private byte[] bytes = new byte[784];

    private final int sizeTrainBatch;
    private final int sizeTestBatch;

    private ArrayList<String> labels;

    public QuickDrawLoader1D() {
        transformData = new TransformData.Sigmoid();
        sizeTestBatch = 100;
        sizeTrainBatch = 2000;
        test = new ArrayList<>(130 * sizeTestBatch);
        train = new ArrayList<>(130 * sizeTrainBatch);

        ArrayList<String> labels = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File("D:/datasets/quick_draw/quick_draw_label.txt"))) {
            while (scanner.hasNextLine()) {
                labels.add(scanner.nextLine());
            }
            loadTestData();
            loadTrainData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public QuickDrawLoader1D(int sizeTrainBatch, int sizeTestBatch) {
        transformData = new TransformData.Sigmoid();
        test = new ArrayList<>(130 * sizeTestBatch);
        train = new ArrayList<>(130 * sizeTrainBatch);
        this.sizeTrainBatch = sizeTrainBatch;
        this.sizeTestBatch = sizeTestBatch;

        labels = new ArrayList<>();
        try (Scanner scanner = new Scanner(new File("D:/datasets/quick_draw/quick_draw_label.txt"))) {
            while (scanner.hasNextLine()) {
                labels.add(scanner.nextLine());
            }
            loadTestData();
            loadTrainData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public QuickDrawLoader1D(int sizeTrainBatch, int sizeTestBatch, String... label) {
        this(sizeTrainBatch, sizeTestBatch, new TransformData.Sigmoid(), label);
    }

    public QuickDrawLoader1D(int sizeTrainBatch, int sizeTestBatch, TransformData transformData, String... label) {
        this.transformData = transformData;
        test = new ArrayList<>(130 * sizeTestBatch);
        train = new ArrayList<>(130 * sizeTrainBatch);
        this.sizeTrainBatch = sizeTrainBatch;
        this.sizeTestBatch = sizeTestBatch;

        labels = new ArrayList<>();
        try {
            Collections.addAll(labels, label);
            loadTestData();
            loadTrainData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
        int index = 0;
        train.removeAll(train);
        for (String label : labels) {
            FileInputStream scanner = new FileInputStream("D:/datasets/quick_draw/full_numpy_bitmap_" + label + ".npy");
            scanner.skip(80);
            scanner.skip(sizeTestBatch * 784);
            for (int i = 0; i < sizeTrainBatch; i++) {
                scanner.read(bytes);
                generateInput();
                NNVector labelClass = new NNVector(labels.size());
                labelClass.set(index, 1);
                train.add(new ImageData1D(new NNVector(inputsData, null), labelClass));
            }
            index++;
        }
        Collections.shuffle(train);
    }

    private void loadTestData() throws IOException {
        test.removeAll(test);
        int index = 0;
        for (String label : labels) {
            FileInputStream scanner = new FileInputStream("D:/datasets/quick_draw/full_numpy_bitmap_" + label + ".npy");
            scanner.skip(80);
            for (int i = 0; i < sizeTestBatch; i++) {
                scanner.read(bytes);
                generateInput();
                NNVector labelClass = new NNVector(labels.size());
                labelClass.set(index, 1);
                test.add(new ImageData1D(new NNVector(inputsData, null), labelClass));
            }
            index++;
        }
        Collections.shuffle(test);
    }

    private void generateInput() {
        inputsData = new float[784];
        for (int i = 0; i < 784; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
    }
}