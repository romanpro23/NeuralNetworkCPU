package test.utilitas;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class RewrtiteLabel {
    public static void main(String[] args) throws IOException {
        Scanner tiny = new Scanner(new File("D:\\datasets\\ImageNet\\label250.txt"));
        FileWriter writer = new FileWriter(new File("D:\\datasets\\ImageNet\\label250_val.txt"));

        for (int i = 0; i < 250; i++) {
            String str = tiny.nextLine();
            Scanner scanner = new Scanner(new File("D:\\datasets\\ImageNet\\words.txt"));
            while(scanner.hasNextLine()){
                String[] label = scanner.nextLine().split("\t");
                if(label[0].equals(str)){
                    writer.write(label[1] + "\n");
                    writer.flush();
                    break;
                }
            }
        }
    }
}
