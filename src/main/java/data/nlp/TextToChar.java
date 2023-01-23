package data.nlp;

import java.util.LinkedHashSet;
import java.util.Scanner;
import java.util.Set;

public class TextToChar {
    public LinkedHashSet<Character> textToChar(Scanner scanner){
        LinkedHashSet<Character> chars = new LinkedHashSet<>();
        while (scanner.hasNext()){
            char[] text = scanner.nextLine().toCharArray();
            for (char c : text) {
                chars.add(c);
            }
        }

        return chars;
    }
}
