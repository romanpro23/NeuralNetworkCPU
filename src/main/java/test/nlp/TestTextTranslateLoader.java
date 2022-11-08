package test.nlp;

import data.nlp.EnUaTranslateLoader;
import data.nlp.TextTranslateLoader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Scanner;

public class TestTextTranslateLoader {
    public static void main(String[] args) throws IOException {
//        TextTranslateLoader loader = new TextTranslateLoader("D:/datasets/nlp/ukr.txt");
//        TextTranslateLoader loader = new TextTranslateLoader("D:/datasets/nlp/en-ua.txt");
//
//        loader.rewrite("D:/datasets/nlp/en-ua.txt");
//        loader.createVocabularyWords("D:/datasets/nlp/");

//        EnUaTranslateLoader loader = new EnUaTranslateLoader(10500, 29000);
//
//        System.out.println(Arrays.toString(loader.getNextTrainData(1).getInput()[0].getData()));

        TextTranslateLoader loader = new TextTranslateLoader();

        loader.rewrite("D:/datasets/flickr/results.txt", "D:/datasets/flickr/result.txt");
        loader.createOneVocabularyWords(new Scanner(new File("D:/datasets/flickr/result.txt")), "D:/datasets/flickr/vocabulary.txt");
    }
}
