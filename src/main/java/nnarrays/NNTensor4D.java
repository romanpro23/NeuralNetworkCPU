package nnarrays;

import lombok.Getter;

public class NNTensor4D extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private final int length;

    private NNTensor[] data;

    public NNTensor4D(int depth, int length, int row, int column) {
        super();
        this.column = column;
        this.row = row;
        this.depth = depth;
        this.length = length;
        data = new NNTensor[depth];
        for (int i = 0; i < depth; i++) {
            data[i] = new NNTensor(length, row, column);
        }
        countAxes = 4;
    }

    @Override
    public int size(){
        return length * depth * row * column;
    }

    public int depth(){
        return depth;
    }

    public NNTensor[] data(){
        return data;
    }

    @Override
    public int[] getSize(){
        return new int[]{depth, length, row, column};
    }
}
