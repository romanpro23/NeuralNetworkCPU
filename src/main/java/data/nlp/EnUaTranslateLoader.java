package data.nlp;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.network_train.NNData1D;
import lombok.SneakyThrows;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

public class EnUaTranslateLoader extends DataLoader1D {
    private final int sizeEnVocabulary;
    private final int sizeUaVocabulary;

    private LinkedHashMap<Integer, String> enWords;
    private LinkedHashMap<Integer, String> uaWords;

    private LinkedHashMap<String, Integer> enVocabulary;
    private LinkedHashMap<String, Integer> uaVocabulary;

    private int maxLength;
    private boolean addPaddingOnStart;

    public EnUaTranslateLoader(int sizeEnVocabulary, int sizeUaVocabulary) {
        this(sizeEnVocabulary, sizeUaVocabulary, 0, false);
    }

    public EnUaTranslateLoader(int sizeEnVocabulary, int sizeUaVocabulary, int maxLength, boolean addPaddingOnStart) {
        this.sizeEnVocabulary = sizeEnVocabulary;
        this.sizeUaVocabulary = sizeUaVocabulary;

        test = new ArrayList<>(0);
        train = new ArrayList<>(25000);

        if(maxLength != 0){
            this.maxLength = maxLength;
            this.addPaddingOnStart = addPaddingOnStart;
        }

        loadData();
    }

    @SneakyThrows
    private void loadData() {
        Scanner scannerEnV = new Scanner(new File("D:/datasets/nlp/ua/en.txt"));
        Scanner scannerUaV = new Scanner(new File("D:/datasets/nlp/ua/ua.txt"));
        Scanner scanner = new Scanner(new File("D:/datasets/nlp/ua/en-ua.txt"));

        enVocabulary = new LinkedHashMap<>();
        uaVocabulary = new LinkedHashMap<>();
        enWords = new LinkedHashMap<>();
        uaWords = new LinkedHashMap<>();

        enVocabulary.put(NLP._SOS, NLP.SOS);
        enWords.put(NLP.SOS, NLP._SOS);
        uaVocabulary.put(NLP._SOS, NLP.SOS);
        uaWords.put(NLP.SOS, NLP._SOS);
        enVocabulary.put(NLP._EOS, NLP.EOS);
        enWords.put(NLP.EOS, NLP._EOS);
        uaVocabulary.put(NLP._EOS, NLP.EOS);
        uaWords.put(NLP.EOS, NLP._EOS);
        enVocabulary.put(NLP._UNK, NLP.UNK);
        enWords.put(NLP.UNK, NLP._UNK);
        uaVocabulary.put(NLP._UNK, NLP.UNK);
        uaWords.put(NLP.UNK, NLP._UNK);
        enVocabulary.put(NLP._PAD, NLP.PAD);
        enWords.put(NLP.PAD, NLP._PAD);
        uaVocabulary.put(NLP._PAD, NLP.PAD);
        uaWords.put(NLP.PAD, NLP._PAD);

        for (int i = 4; i < sizeEnVocabulary; i++) {
            String str = scannerEnV.nextLine();
            enVocabulary.put(str, i);
            enWords.put(i, str);
        }
        for (int i = 4; i < sizeUaVocabulary; i++) {
            String str = scannerUaV.nextLine();
            uaVocabulary.put(str, i);
            uaWords.put(i, str);
        }

        while (scanner.hasNextLine()) {
            String[] data = scanner.nextLine().split("\t");
            String[] enWords = data[0].split(" ");
            String[] uaWords = data[1].split(" ");

            NNVector input = getEnVector(enWords);
            NNVector output = getUaVector(uaWords);

            train.add(new ImageData1D(input, output));
        }

        Collections.shuffle(train);
    }

    public NNVector getUaVector(String[] uaWords){
        NNVector input;
        int index;
        int start = 1;
        int end = uaWords.length;
        if(maxLength != 0 ){
            if(addPaddingOnStart) {
                start = maxLength - uaWords.length - 1;
                end = maxLength - 1;
                input = new NNVector(maxLength);
                input.set(0, NLP.SOS);
                for (int i = 1; i < start; i++) {
                    input.set(i, NLP.PAD);
                }
            } else {
                end = maxLength - uaWords.length - 1;
                input = new NNVector(maxLength);
                input.set(end, NLP.SOS);
                for (int i = end + 1; i < maxLength; i++) {
                    input.set(i, NLP.PAD);
                }
            }
        } else {
            input = new NNVector(uaWords.length + 2);
            input.set(0, NLP.SOS);
        }

        for (int i = start; i < end; i++) {
            Integer data = uaVocabulary.get(uaWords[i]);
            index = Objects.requireNonNullElse(data, NLP.UNK);
            input.set(i, index);
        }
        input.set(end + 1, NLP.EOS);

        return input;
    }

    public NNVector getEnVector(String[] enWords){
        NNVector input;
        int index;
        int start = 1;
        int end = enWords.length;
        if(maxLength != 0 ){
            if(addPaddingOnStart) {
                start = maxLength - enWords.length - 1;
                end = maxLength - 1;
                input = new NNVector(maxLength);
                input.set(0, NLP.SOS);
                for (int i = 1; i < start; i++) {
                    input.set(i, NLP.PAD);
                }
            } else {
                end = maxLength - enWords.length - 1;
                input = new NNVector(maxLength);
                input.set(end, NLP.SOS);
                for (int i = end + 1; i < maxLength; i++) {
                    input.set(i, NLP.PAD);
                }
            }
        } else {
            input = new NNVector(enWords.length + 2);
            input.set(0, NLP.SOS);
        }

        for (int i = start; i < end; i++) {
            Integer data = enVocabulary.get(enWords[i]);
            index = Objects.requireNonNullElse(data, NLP.UNK);
            input.set(i, index);
        }
        input.set(end + 1, NLP.EOS);

        return input;
    }

    public String decodeUaString(NNVector input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == NLP.EOS) {
                break;
            }
            string.append(uaWords.get((int) input.get(i)) + " ");
        }
        return string.toString();
    }

    public String decodeEnString(NNVector input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == NLP.EOS) {
                break;
            }
            string.append(enWords.get((int) input.get(i)) + " ");
        }
        return string.toString();
    }
}
