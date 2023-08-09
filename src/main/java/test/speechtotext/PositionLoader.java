package test.speechtotext;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.nlp.NLP;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class PositionLoader extends DataLoader3D {
    private LinkedHashMap<Integer, Character> uaChars;
    private LinkedHashMap<Character, Integer> codeUaChars;

    private int sizeBuffer = 5000;

    public PositionLoader(int countChars) throws Exception {
        uaChars = new LinkedHashMap<>();
        codeUaChars = new LinkedHashMap<>();
        train = new ArrayList<>();
        test = new ArrayList<>();

        try {
            Scanner scanner = new Scanner(new File("C:/Levani/NeuralNetwork/data/ka_chars.txt"), StandardCharsets.UTF_8);
            //uaChars.put(NLP.UNK, NLP._UNK_CHAR);
            //codeUaChars.put(NLP._UNK_CHAR, NLP.UNK);
            uaChars.put(NLP.PAD, ' ');
            codeUaChars.put(' ', NLP.PAD);

            int key = 1;
            for (int i = key; i < countChars; i++) {
                Character str = scanner.nextLine().charAt(0);
                uaChars.put(key, str);
                codeUaChars.put(str, key);
                key++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        reloadTrainData();
    }

    @Override
    public void reloadTrainData() {

        String fileName = "C:/Levani/cv-corpus-14.0-2023-06-23/ka/validated.tsv";
        File file = new File(fileName);
        ArrayList<String[]> data = tsvr(file);

        File folder = new File("C:/tesstrain-windows-gui-main/SpeechToSpeech-0.012/images/");
        File[] listOfFiles = folder.listFiles();

        int wwq = 0;
        for (File afile : listOfFiles) {
            if (afile.isFile()) {

                BufferedImage img = null;
                try {
                    img = ImageIO.read(new File(folder.toPath() + "\\" + afile.getName()));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                int width = img.getWidth();
                int height = img.getHeight();

                int size_width = 488;

                if (width > size_width) {
                    try {
                        throw new Exception("!!!: " + width);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }

                int ss = 0;
                float[] inputsData = new float[size_width * 24];
                for (int xx = 0; xx < size_width; xx++) {
                    if (width > xx) {
                        for (int yy = 0; yy < height; yy++) {
                            Color originalColor = new Color(img.getRGB(xx, yy));
                            int pixels = originalColor.getRed();

                            inputsData[ss] = (float) pixels;
                            ss++;
                        }
                    } else {
                        for (int yy = 0; yy < height; yy++) {
                            inputsData[ss] = 0.0f;
                            ss++;
                        }
                    }
                }

                var label = "";
                for (String[] datum : data) {
                    if (Objects.equals(fileNameWithOutExt(datum[1]), fileNameWithOutExt(afile.getName()))) {
                        label = datum[2];
                        break;
                    }
                }

                if (!Objects.equals(label, "")) {

                    int sss = 175 - label.length();
                    if (sss < 0) {
                        try {
                            throw new Exception("!!!: " + sss);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }
                    for (int ww = 0; ww < sss; ww++) {
                        label += " ";
                    }

                    NNVector output = codeString(label);

                    train.add(new ImageData3D(new NNTensor(size_width, 24, 1, inputsData), output));
                    test.add(new ImageData3D(new NNTensor(size_width, 24, 1, inputsData), output));

                    if (wwq % 100 == 0) {
                        System.out.println(wwq);
                    }

                    if (wwq == 500) {
                        return;
                    }

                    wwq++;
                }
            }
        }
    }

    public NNVector codeString(String text) {
        char[] chars = text.toCharArray();
        NNVector input = new NNVector(chars.length);
        for (int j = 0; j < input.size(); j++) {
            float value = ((float) codeUaChars.get(chars[j]));
            input.set(j, value);
        }

        return input;
    }

    public String decodeString(float[] input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.length; i++) {
            Character Char = (uaChars.get((int) (Math.round(input[i])/* uaChars.size()*/)));
            if (Char != null) {
                string.append(Char);
            }
        }
        return string.toString();
    }

    public double cosineSimilarity(float[] vectorA, float[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    public static String[][] ClockWiseRotation(String[] data, int StartPos, int column, int row) {
        int ColumnLeng = (data.length - StartPos) / row;
        String[][] a = new String[ColumnLeng][row];
        int dd = StartPos;
        for (int v = 0; v < ColumnLeng; v++) {
            for (int j = 0; j < row; j++) {
                a[v][j] = data[dd];
                dd++;
            }
        }

        return resize(rotateMatrixBy90DegreeClockwise(a), column, row);
    }

    static String[][] resize(String[][] matrix, int w, int h) {
        String[][] temp = new String[h][w];
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[0].length; j++) {
                temp[i][j] = "0.1";
            }
        }

        h = Math.min(h, matrix.length);
        w = Math.min(w, matrix[0].length);
        for (int i = 0; i < h; i++)
            System.arraycopy(matrix[i], 0, temp[i], 0, w);
        return temp;
    }

    private static String[][] rotateMatrixBy90DegreeClockwise(String[][] matrix) {
        int totalRowsOfRotatedMatrix = matrix[0].length; //Total columns of Original Matrix
        int totalColsOfRotatedMatrix = matrix.length; //Total rows of Original Matrix

        String[][] rotatedMatrix = new String[totalRowsOfRotatedMatrix][totalColsOfRotatedMatrix];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                rotatedMatrix[j][(totalColsOfRotatedMatrix - 1) - i] = matrix[i][j];
            }
        }
        return rotatedMatrix;
    }

    private static Double convertToDouble(String item) {
        double value = 0;
        try {
            value = Double.parseDouble(item);
        } catch (NumberFormatException numberFormatException) {
        }
        return value;
    }

    public static ArrayList<String[]> tsvr(File test2) {
        ArrayList<String[]> Data = new ArrayList<>(); //initializing a new ArrayList out of String[]'s
        try (BufferedReader TSVReader = new BufferedReader(new InputStreamReader(new FileInputStream(test2), "UTF-8"))) {
            String line = null;
            while ((line = TSVReader.readLine()) != null) {
                String[] lineItems = line.split("\t"); //splitting the line and adding its items in String[]
                Data.add(lineItems); //adding the splitted line array to the ArrayList
            }
        } catch (Exception e) {
            System.out.println("Something went wrong");
        }
        return Data;
    }

    public static String fileNameWithOutExt(String fileName) {
        return Optional.of(fileName.lastIndexOf(".")).filter(i -> i >= 0)
                .filter(i -> i > fileName.lastIndexOf(File.separator))
                .map(i -> fileName.substring(0, i)).orElse(fileName);
    }
}