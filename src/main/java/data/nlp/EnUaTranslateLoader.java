package data.nlp;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.network_train.NNData1D;
import lombok.SneakyThrows;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Scanner;

public class EnUaTranslateLoader extends DataLoader1D {
    private int sizeEnVocabulary, sizeUaVocabulary;

    public EnUaTranslateLoader(int sizeEnVocabulary, int sizeUaVocabulary) {
        this.sizeEnVocabulary = sizeEnVocabulary;
        this.sizeUaVocabulary = sizeUaVocabulary;

        test = new ArrayList<>(0);
        train = new ArrayList<>(25000);

        loadData();
    }

    @SneakyThrows
    private void loadData(){
        Scanner scannerEnV = new Scanner(new File("D:/datasets/nlp/ua/en.txt"));
        Scanner scannerUaV = new Scanner(new File("D:/datasets/nlp/ua/ua.txt"));
        Scanner scanner = new Scanner(new File("D:/datasets/nlp/ua/en-ua.txt"));

        LinkedHashMap<String, Integer> enVocabulary = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> uaVocabulary = new LinkedHashMap<>();
        enVocabulary.put("<START>", 0);
        uaVocabulary.put("<START>", 0);
        enVocabulary.put("<END>", 1);
        enVocabulary.put("<UNK>", 2);
        uaVocabulary.put("<END>", 1);
        uaVocabulary.put("<UNK>", 2);

        for (int i = 3; i < sizeEnVocabulary; i++) {
            enVocabulary.put(scannerEnV.nextLine(), i);
        }
        for (int i = 3; i < sizeUaVocabulary; i++) {
            uaVocabulary.put(scannerUaV.nextLine(), i);
        }

        while (scanner.hasNextLine()){
            String[] data = scanner.nextLine().split("\t");
            String[] enWords = data[0].split(" ");
            String[] uaWords = data[0].split(" ");

            NNVector input = new NNVector(enWords.length + 1);
            int index;
            for (int i = 0; i < enWords.length; i++) {
                if(enVocabulary.get(enWords[i]) != null){
                    index = enVocabulary.get(enWords[i]);
                } else {
                    index = 2;
                }
                input.set(i + 1, index);
            }

            NNVector output = new NNVector(uaWords.length + 1);
            for (int i = 0; i < uaWords.length; i++) {
                if(uaVocabulary.get(enWords[i]) != null){
                    index = uaVocabulary.get(enWords[i]);
                } else {
                    index = 2;
                }
                output.set(i + 1, index);
            }

            train.add(new ImageData1D(input, output));
        }

        Collections.shuffle(train);
    }
}
