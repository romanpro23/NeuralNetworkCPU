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

    public NNTensor4D(int depth, int length, int row, int column) {
        super(depth*length*row*column);
        this.column = column;
        this.row = row;
        this.depth = depth;
        this.length = length;

        countAxes = 4;
    }

    @Override
    public int size(){
        return length * depth * row * column;
    }

    public int depth(){
        return depth;
    }

    @Override
    public int[] getSize(){
        return new int[]{depth, length, row, column};
    }
}
