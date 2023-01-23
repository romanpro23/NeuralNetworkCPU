package data.nlp;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class UaTextRedactor {
    public void rewrite(Scanner scanner, String path) throws IOException {
        String text;

        FileWriter writer = new FileWriter(path);
        int index = 0;
        while (scanner.hasNextLine()) {
            index++;
            if(index % 10000 == 0){
                System.out.println(index);
            }
            text = scanner.nextLine().toLowerCase();
            for (int i = 0; i < text.length(); i++) {
                char ch = text.charAt(i);
                if((ch >= 1072 && ch <= 1103) ||(ch >= 1108 && ch <= 1111) || (ch >= '0' && ch <= '9')  || ch == '!'
                        || ch == ' ' || ch == '\'' || ch == '-' || ch == ',' || ch == '.' || ch == '?'){
                    writer.write(ch);
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }
}
