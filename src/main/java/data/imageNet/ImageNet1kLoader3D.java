package data.imageNet;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class ImageNet1kLoader3D extends DataLoader3D {
    private static final int sizeImage = 64;

    private TransformData transformData;

    private ArrayList<String> labels;
    private int sizeLabelsT[];
    private int nT[];
    private int sizeLabelsV[];
    private int nV[];

    public ImageNet1kLoader3D() {
        this(new TransformData.Sigmoid());
    }

    public ImageNet1kLoader3D(TransformData transformData) {
        this.transformData = transformData;

        test = new ArrayList<>(1600);
        train = new ArrayList<>(6400);
        labels = new ArrayList<>();
        sizeLabelsV = new int[1000];
        sizeLabelsT = new int[1000];
        nT = new int[1000];
        nV = new int[1000];

        try {
            Scanner reader = new Scanner(new File("D:\\datasets\\ImageNet\\labels1k.txt"));
            int i = 0;
            while (reader.hasNextLine()){
                String lbl = reader.nextLine();
                labels.add(lbl);

                File fT = new File("D:\\datasets\\ImageNet\\train\\" + lbl + "\\");
                File fV = new File("D:\\datasets\\ImageNet\\valid\\" + lbl + "\\");

                sizeLabelsV[i] = fV.listFiles().length;
                sizeLabelsT[i] = fT.listFiles().length;

                nT[i] = (int) (Math.random() * sizeLabelsT[i]);
                nV[i] = (int) (Math.random() * sizeLabelsV[i]);

                i++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


        loadData();
    }

    public ImageNet1kLoader3D setTransformData(TransformData transformData){
        this.transformData = transformData;

        return this;
    }

    @SneakyThrows
    private void loadData(){
        reloadTrainData();
        reloadTestData();

        Collections.shuffle(test);
        Collections.shuffle(train);
    }

    @SneakyThrows
    @Override
    protected void reloadTestData(){
        test.removeAll(test);
        for (int i = 0; i < 1000; i++) {
            NNVector output = new NNVector(1000);
            output.set(i, 1);
            for (int j = 0; j < 2; j++) {
                if(nV[i] >= sizeLabelsV[i]){
                    nV[i] = 0;
                }
                test.add(
                        new ImageData3D(
                                loadImage("D:\\datasets\\ImageNet\\valid\\" + labels.get(i) + "\\" + nV[i] + ".png"),
                                output
                        )
                );
                nV[i]++;
            }
        }
    }

    @SneakyThrows
    @Override
    protected void reloadTrainData(){
        train.removeAll(train);
        for (int i = 0; i < 1000; i++) {
            NNVector output = new NNVector(1000);
            output.set(i, 1);
            for (int j = 0; j < 6; j++) {
                if(nT[i] >= sizeLabelsT[i]){
                    nT[i] = 0;
                }
                train.add(
                        new ImageData3D(
                                loadImage("D:\\datasets\\ImageNet\\train\\" + labels.get(i) + "\\" + nT[i] + ".png"),
                                output
                        )
                );
                nT[i]++;
            }
        }
    }

    public ImageNet1kLoader3D useCrop(){
        this.useCrop = true;

        return this;
    }

    public ImageNet1kLoader3D useReverse(){
        this.useReverse = true;

        return this;
    }

    @SneakyThrows
    private  NNTensor loadImage(String path){
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File(path));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                input.set(i, j, 0, transformData.transformR(color.getRed()));
                input.set(i, j, 1, transformData.transformG(color.getGreen()));
                input.set(i, j, 2, transformData.transformB(color.getBlue()));
            }
        }
        return input;
    }

    @SneakyThrows
    public static String[] getLabelVal(int... val){
        Scanner readerVal = new Scanner(new File("D:\\datasets\\ImageNet\\labels1k_val.txt"));
        ArrayList<String> labelsVal = new ArrayList<>(1000);
        while(readerVal.hasNextLine()){
            labelsVal.add(readerVal.nextLine().split(",")[0]);
        }

        String[] result = new String[val.length];
        for (int i = 0; i < val.length; i++) {
            result[i] = labelsVal.get(val[i]);
        }

        return result;
    }
}
