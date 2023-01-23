package test.utilitas;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Imagenet500Test {
    public static void main(String[] args) throws IOException {
        Scanner reader = new Scanner(new File("D:\\datasets\\ImageNet\\labels1k.txt"));
        FileWriter writer = new FileWriter(new File("D:\\datasets\\ImageNet\\labels500.txt"));
        for (int i = 0; i < 500; i++) {
            String str1 = reader.nextLine();
            String str2 = reader.nextLine();

            if(Math.random() < 0.5){
                writer.write(str1 + "\n");
                writer.flush();
            } else {
                writer.write(str2 + "\n");
                writer.flush();
            }
        }
    }
}
