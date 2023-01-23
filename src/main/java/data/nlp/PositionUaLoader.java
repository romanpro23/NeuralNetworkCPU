package data.nlp;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import lombok.SneakyThrows;
import nnarrays.NNVector;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class PositionUaLoader extends DataLoader1D {
    private Scanner scanner;

    private LinkedHashMap<Integer, Character> uaChars;
    private LinkedHashMap<Character, Integer> codeUaChars;

    private int sizeBuffer = 5000;

    public PositionUaLoader(int countChars) {
        loadScanner();
        uaChars = new LinkedHashMap<>();
        codeUaChars = new LinkedHashMap<>();
        train = new ArrayList<>();
        test = new ArrayList<>();

        try {
            Scanner scanner = new Scanner(new File("D:\\datasets\\nlp\\Ukrainian\\ua_chars.txt"));
            uaChars.put(NLP.UNK, NLP._UNK_CHAR);
            codeUaChars.put(NLP._UNK_CHAR, NLP.UNK);
            uaChars.put(NLP.PAD, ' ');
            codeUaChars.put(' ', NLP.PAD);

            int key = 2;
            for (int i = key; i < countChars; i++) {
                Character str = scanner.nextLine().charAt(0);
                uaChars.put(key, str);
                codeUaChars.put(str, key);
                key++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        int rand = (int) (Math.random() * 1800000);
        for (int i = 0; i < rand; i++) {
            scanner.nextLine();
        }
        reloadTrainData();
    }

    @SneakyThrows
    private void loadScanner() {
        scanner = new Scanner(new File("D:\\datasets\\nlp\\Ukrainian\\ua_fiction.txt"));
    }

    @Override
    public void reloadTrainData() {
        train.removeAll(train);
        for (int i = 0; i < sizeBuffer; i++) {
            if (scanner.hasNextLine()) {
                String str = scanner.nextLine();
                if(str.length() > 1) {
                    NNVector input = codeString(str);
                    train.add(new ImageData1D(input, input));
                }
            } else {
                loadScanner();
            }
        }
        Collections.shuffle(train);
    }

    public NNVector codeString(String text) {
        char[] chars = text.toCharArray();
        NNVector input = new NNVector(chars.length);
        for (int j = 0; j < input.size(); j++) {
            Integer value = codeUaChars.get(chars[j]);
            input.set(j, Objects.requireNonNullElse(value, NLP.UNK));
        }

        return input;
    }

    public String decodeString(NNVector input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.size(); i++) {
            string.append(uaChars.get((int) input.get(i)));
        }
        return string.toString();
    }
}
