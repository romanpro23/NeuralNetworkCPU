//package data.nlp;
//
//import com.opencsv.CSVReader;
//import com.opencsv.exceptions.CsvValidationException;
//
//import java.io.File;
//import java.io.FileReader;
//import java.io.FileWriter;
//import java.io.IOException;
//import java.util.*;
//
//public class IMDBLoader extends TextTranslateLoader {
//    private CSVReader reader;
//    private int sizeVocabulary;
//    private int countOftenWord;
//    private int maxLength;
//
//    private int indexTest = 25000;
//    private int indexTrain = 0;
//
//    private ArrayList<Tensor> data;
//    private ArrayList<Vector> key;
//    private ArrayList<String> words;
//
//    private LinkedHashMap<String, Integer> map;
//
//    public IMDBLoader(int sizeVocabulary, int countOftenWord, int maxLength) {
//        this.countOftenWord = countOftenWord;
//        this.maxLength = maxLength;
//        this.sizeVocabulary = sizeVocabulary - 2;
//        words = new ArrayList<>();
//        data = new ArrayList<>();
//        key = new ArrayList<>();
//        map = new LinkedHashMap<>(10000);
//        try {
//            reader = new CSVReader(new FileReader("src/main/resources/text/IMDB Dataset.csv"));
//            loadData();
//        } catch (IOException | CsvValidationException e) {
//            e.printStackTrace();
//        }
//    }
//
//    private void loadData() throws CsvValidationException, IOException {
//        String[] line;
//        reader.skip(1);
//
//        ArrayList<ArrayList<String>> reviews = new ArrayList<>();
//        for (int i = 0; i < 50000; i++) {
//            line = reader.readNext();
//            reviews.add(StringRedactor.slemming(StringRedactor.deleteStopWord(StringRedactor.generateString(line[0]))));
//
//            if (line[1].equals("positive")) {
//                key.add(new Vector(new float[]{1}));
//            } else {
//                key.add(new Vector(new float[]{0}));
//            }
//
////            for (int j = 0; j < reviews.get(i).size(); j++) {
////                if (map.containsKey(reviews.get(i).get(j))) {
////                    Integer val = map.get(reviews.get(i).get(j));
////                    val++;
////                    map.put(reviews.get(i).get(j), val);
////                } else {
////                    map.put(reviews.get(i).get(j), 1);
////                }
////            }
//        }
////        map = sortHashMapByValues(map);
////        Iterator<Map.Entry<String, Integer>> entries = map.entrySet().iterator();
////        loadDataWords(entries);
//
//        ArrayList<String> temp = new ArrayList<>();
//        Scanner reader = new Scanner(new File("src/main/resources/text/words.txt"));
//        while (reader.hasNextLine()) {
//            temp.add(reader.nextLine());
//        }
//
//        for (int i = temp.size() - 1 - countOftenWord; i >= temp.size() - countOftenWord - sizeVocabulary; i--) {
//            words.add(temp.get(i));
//        }
//
//        generateInputs(reviews);
//        map = null;
//        words = null;
//    }
//
//    private void loadDataWords(Iterator<Map.Entry<String, Integer>> entries) throws IOException {
//        FileWriter writer = new FileWriter("src/main/resources/text/words.txt");
//        while (entries.hasNext()) {
//            Map.Entry<String, Integer> entry = entries.next();
//            writer.write(entry.getKey() + "\n");
//            writer.flush();
//        }
//    }
//
//    private void generateInputs(ArrayList<ArrayList<String>> review) {
//        int index;
//        String word;
//        for (int i = 0; i < review.size(); i++) {
//            data.add(new Tensor(1, 1, maxLength));
//            for (int j = 0; j < maxLength; j++) {
//                index = 1;
//                if (review.get(i).size() - 1 - j < 0) {
//                    break;
//                }
//                word = review.get(i).get(review.get(i).size() - 1 - j);
//                for (int k = 0; k < words.size(); k++) {
//                    if (word.equals(words.get(k))) {
//                        index += k;
//                        index++;
//                    }
//                }
//
//                data.get(i).getData()[0][0][maxLength - 1 - j] = index;
//            }
//        }
//    }
//
//    private String formatWord(String key) {
//        if (key.endsWith("<br")) {
//            key = key.substring(0, key.length() - 3);
//        }
//        if (key.startsWith("/>")) {
//            key = key.substring(2);
//        }
//        while (key.endsWith(".") || key.endsWith(",") || key.endsWith("!") || key.endsWith("-") || key.endsWith("&")
//                || key.endsWith(":") || key.endsWith(";") || key.endsWith("?") || key.endsWith("\'")
//                || key.endsWith("\"") || key.endsWith(")") || key.endsWith("}") || key.endsWith("]")
//                || key.endsWith("*") || key.endsWith("\\") || key.endsWith("/") || key.endsWith("$")
//                || key.endsWith("(") || key.endsWith("+") || key.endsWith("=") || key.endsWith("{") || key.endsWith("[")
//                || key.endsWith("\u0085") || key.endsWith("@") || key.endsWith("#") || key.endsWith("%")
//                || key.endsWith("\t") || key.endsWith("^") || key.endsWith("\u0010") || key.endsWith("<")
//                || key.endsWith(">") || key.endsWith("`")) {
//            key = key.substring(0, key.length() - 1);
//        }
//        while (key.startsWith("-") || key.startsWith("{") || key.startsWith("(") || key.startsWith("[")
//                || key.startsWith(":") || key.startsWith("\'") || key.startsWith("\"") || key.startsWith("\\")
//                || key.startsWith("/") || key.startsWith("*") || key.startsWith("$") || key.startsWith("@")
//                || key.startsWith(">") || key.startsWith("\t") || key.startsWith("\b") || key.startsWith("#")
//                || key.startsWith("%") || key.startsWith("^") || key.startsWith("&") || key.startsWith("+")
//                || key.startsWith(")") || key.startsWith("}") || key.startsWith("]") || key.startsWith("\u0085")
//                || key.startsWith("\u0010") || key.startsWith("<") || key.startsWith("=") || key.startsWith("`")) {
//            key = key.substring(1);
//        }
//        return key;
//    }
//
//    public LinkedHashMap<String, Integer> sortHashMapByValues(HashMap<String, Integer> passedMap) {
//        List<String> mapKeys = new ArrayList<>(passedMap.keySet());
//        List<Integer> mapValues = new ArrayList<>(passedMap.values());
//        Collections.sort(mapValues);
//        Collections.sort(mapKeys);
//
//        LinkedHashMap<String, Integer> sortedMap = new LinkedHashMap<>();
//        Iterator<Integer> valueIt = mapValues.iterator();
//        int i = 0;
//        while (valueIt.hasNext()) {
//            System.out.println(i++);
//            Integer val = valueIt.next();
//            Iterator<String> keyIt = mapKeys.iterator();
//
//            while (keyIt.hasNext()) {
//                String key = keyIt.next();
//                Integer comp1 = passedMap.get(key);
//                Integer comp2 = val;
//
//                if (comp1.equals(comp2)) {
//                    keyIt.remove();
//                    sortedMap.put(key, val);
//                    break;
//                }
//            }
//        }
//        return sortedMap;
//    }
//
//    @Override
//    public TextData getNextTestData() {
//        if (indexTrain > 25000) {
//            indexTrain = 0;
//        }
//        TextData text = new TextData(data.get(indexTrain), key.get(indexTrain));
//        indexTrain++;
//        return text;
//    }
//
//    @Override
//    public TextData getNextTrainData() {
//        if (indexTest >= 50000) {
//            indexTest = 25000;
//        }
//        TextData text = new TextData(data.get(indexTest), key.get(indexTest));
//        indexTest++;
//        return text;
//    }
//}
//
//
