package utilities;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.awt.event.HierarchyEvent;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import nnarrays.NNMatrix;
import org.jblas.FloatMatrix;
import org.jblas.Solve;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcublas.cublasAtomicsMode;
import jcuda.jcublas.cublasPointerMode;
import jcuda.jcublas.cublasSideMode;
import jcuda.runtime.JCuda;

public class CublasUtil {

    public static final boolean DEBUG_SYNC = false;

    public static LinkedList<Matrix> allocated;
    public static cublasHandle cublasHandle;
    public static CUmodule helperModule;

    public static void startup() {
        JCudaHelper.init();
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
        helperModule = JCudaHelper.compile("la_helper_funs", Matrix.kernels);
        allocated = new LinkedList<Matrix>();
        JCublas2.cublasSetAtomicsMode(cublasHandle, cublasAtomicsMode.CUBLAS_ATOMICS_ALLOWED);
        JCublas2.cublasSetPointerMode(cublasHandle, cublasPointerMode.CUBLAS_POINTER_MODE_HOST);
    }

    public static void shutdown() {
        if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        freeAll(true);
        JCublas2.cublasDestroy(cublasHandle);
        //CudaUtil.shutdown();
    }

    public static void freeAll() {
        freeAll(false);
    }

    public static void freeAll(boolean freeDontFree) {
        if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        LinkedList<Matrix> remainingAllocated = new LinkedList<Matrix>();
        while (!allocated.isEmpty()) {
            Matrix mat = allocated.poll();
            if (freeDontFree || !mat.dontFree) {
                mat.free();
            } else {
                remainingAllocated.add(mat);
            }
        }
        allocated = remainingAllocated;
    }

    public static void freeAllBut(Matrix... args) {
        Collection<Matrix> keep = new HashSet<Matrix>();
        for (Matrix mat : args) keep.add(mat);
        freeAllBut(keep);
    }

    public static void freeAllBut(Collection<Matrix> keep) {
        if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        LinkedList<Matrix> remainingAllocated = new LinkedList<Matrix>();
        while (!allocated.isEmpty()) {
            Matrix mat = allocated.poll();
            if (!keep.contains(mat) && !mat.dontFree) {
                mat.free();
            } else {
                remainingAllocated.add(mat);
            }
        }
        allocated = remainingAllocated;
    }

    public static class Matrix {

        private boolean dontFree;
        private int rows;
        private int cols;
        private Pointer data_d;

        public Matrix(int rows, int cols) {
            this.dontFree = false;
            this.rows = rows;
            this.cols = cols;
            this.data_d = new Pointer();
            cudaMalloc(data_d, rows * cols * Sizeof.FLOAT);
            CublasUtil.allocated.add(this);
        }

        public void setDontFree(boolean dontFree) {
            this.dontFree = dontFree;
        }

        public boolean dontFree() {
            return dontFree;
        }

        public boolean equals(Object other) {
            if (other instanceof Matrix) {
                Matrix that = (Matrix) other;
                if (!this.data_d.equals(that.data_d)) {
                    return false;
                } else {
                    return true;
                }
            } else {
                return false;
            }
        }

        public int hashCode() {
            return this.data_d.hashCode();
        }

        public static Matrix build(float[][] mat) {
            Matrix result = new Matrix(mat.length, mat[0].length);
            float[] data_h = toColMajor(mat);
            JCublas2.cublasSetMatrix(result.rows, result.cols, Sizeof.FLOAT, Pointer.to(data_h), result.rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public static Matrix build(int rows, int cols, float[] data_h) {
            Matrix result = new Matrix(rows, cols);

            //cudaMalloc(result.data_d, Sizeof.FLOAT * rows * cols);
            //cudaMemcpy(result.data_d, Pointer.to(data_h), Sizeof.FLOAT * rows * cols, cudaMemcpyHostToDevice);

            JCublas2.cublasSetMatrix(result.rows, result.cols, Sizeof.FLOAT, Pointer.to(data_h), result.rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public static Matrix rand(int rows, int cols, Random rand) {
            return Matrix.build(a.randFloat(rows, cols, rand));
        }

        public static Matrix ones(int rows, int cols) {
            Matrix result = new Matrix(rows, cols);
            result.set(1.0f);
            return result;
        }

        public static Matrix zeros(int rows, int cols) {
            Matrix result = new Matrix(rows, cols);
            result.zeroi();
            return result;
        }

        public static Matrix eye(int n) {
            Matrix result = zeros(n, n);
            result.diagAddi(1.0f);
            return result;
        }

        public boolean isVector() {
            return rows == 1 || cols == 1;
        }

        public boolean isScalar() {
            return rows == 1 && cols == 1;
        }

        public int rows() {
            return rows;
        }

        public int cols() {
            return cols;
        }

        public Matrix copy() {
            Matrix result = new Matrix(rows, cols);
            JCublas2.cublasScopy(cublasHandle, this.rows * this.cols, this.data_d, 1, result.data_d, 1);
//			JCublas2.cublasSetMatrix(result.rows, result.cols, Sizeof.FLOAT, this.data_d, result.rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix copySubmatrix(int r0, int r1, int c0, int c1) {
            Matrix result = new Matrix(r1 - r0, c1 - c0);
            JCublas2.cublasSetMatrix(result.rows, result.cols, Sizeof.FLOAT, this.data_d.withByteOffset((c0 * this.rows + r0) * Sizeof.FLOAT), this.rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix setSubmatrix(int r, int c, Matrix that, int r0, int r1, int c0, int c1) {
            JCublas2.cublasSetMatrix(r1 - r0, c1 - c0, Sizeof.FLOAT, that.data_d.withByteOffset((c0 * that.rows + r0) * Sizeof.FLOAT), that.rows, this.data_d.withByteOffset((c * this.rows + r) * Sizeof.FLOAT), this.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public Matrix setSubmatrix(Matrix that, int r, int c) {
            JCublas2.cublasSetMatrix(that.rows, that.cols, Sizeof.FLOAT, that.data_d, that.rows, this.data_d.withByteOffset((c * this.rows + r) * Sizeof.FLOAT), this.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public Matrix setSubmatrix(float[][] mat, int r, int c) {
            float[] data_h = toColMajor(mat);
            JCublas2.cublasSetMatrix(mat.length, mat[0].length, Sizeof.FLOAT, Pointer.to(data_h), mat.length, this.data_d.withByteOffset((c * this.rows + r) * Sizeof.FLOAT), this.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public Matrix set(int r, int c, float alpha) {
            return setSubmatrix(new float[][]{{alpha}}, r, c);
        }

        public Matrix copyRow(int r) {
            return copySubmatrix(r, r + 1, 0, cols);
        }

        public Matrix copyCol(int c) {
            return copySubmatrix(0, rows, c, c + 1);
        }

        public Matrix setRow(int r, Matrix row) {
            return setSubmatrix(row, r, 0);
        }

        public Matrix setCol(int c, Matrix col) {
            return setSubmatrix(col, 0, c);
        }

        public Matrix set(float alpha) {
            scalarSet(this, alpha);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public float[] toArray() {
            float[] data_h = new float[rows * cols];
            JCublas2.cublasGetVector(data_h.length, Sizeof.FLOAT, data_d, 1, Pointer.to(data_h), 1);
            //          JCublas2.cublasGetMatrix(rows, cols, Sizeof.FLOAT, data_d, rows, Pointer.to(data_h), rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return data_h;
        }

        public float[][] toArray2() {
            float[] data_h = toArray();
            return fromColMajor(data_h, rows);
        }

        public void free() {
            setDontFree(false);
            if (data_d != null) JCuda.cudaFree(data_d);
        }

        //////////////////////////////////////////

        public Matrix diagAdd(float alpha) {
            Matrix diag = new Matrix(1, this.cols);
            diag.set(alpha);
            return diagAdd(diag);
        }

        public Matrix diagAddi(float alpha) {
            Matrix diag = new Matrix(1, this.cols);
            diag.set(alpha);
            return diagAddi(diag);
        }

        public Matrix diagAdd(Matrix diag) {
            Matrix result = this.copy();
            return result.diagAddi(diag);
        }

        public Matrix diagAddi(Matrix diag) {
            JCublas2.cublasSaxpy(cublasHandle, diag.rows * diag.cols, Pointer.to(new float[]{1.0f}), diag.data_d, 1, this.data_d, this.rows + 1);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public Matrix rowMul(Matrix row) {
            Matrix result = new Matrix(this.rows, this.cols);
            dgmm(this, row, result, false);
            return result;
        }

        public Matrix colMul(Matrix col) {
            Matrix result = new Matrix(this.rows, this.cols);
            dgmm(this, col, result, true);
            return result;
        }

        public Matrix rowMuli(Matrix row) {
            dgmm(this, row, this, false);
            return this;
        }

        public Matrix colMuli(Matrix col) {
            dgmm(this, col, this, true);
            return this;
        }

        public Matrix rowDiv(Matrix row) {
            Matrix result = new Matrix(this.rows, this.cols);
            Matrix inverseRow = Matrix.ones(row.rows, row.cols);
            inverseRow.divi(row);
            dgmm(this, inverseRow, result, false);
            return result;
        }

        public Matrix colDiv(Matrix col) {
            Matrix result = new Matrix(this.rows, this.cols);
            Matrix inverseCol = Matrix.ones(col.rows, col.cols);
            inverseCol.divi(col);
            dgmm(this, inverseCol, result, true);
            return result;
        }

        public Matrix rowDivi(Matrix row) {
            Matrix inverseRow = Matrix.ones(row.rows, row.cols);
            inverseRow.divi(row);
            dgmm(this, inverseRow, this, false);
            return this;
        }

        public Matrix colDivi(Matrix col) {
            Matrix inverseCol = Matrix.ones(col.rows, col.cols);
            inverseCol.divi(col);
            dgmm(this, inverseCol, this, true);
            return this;
        }

        public Matrix rowAdd(Matrix row) {
            return this.rowComb(1.0f, row);
        }

        public Matrix rowAddi(Matrix row) {
            return this.rowCombi(1.0f, row);
        }

        public Matrix rowSub(Matrix row) {
            return this.rowComb(-1.0f, row);
        }

        public Matrix rowSubi(Matrix row) {
            return this.rowCombi(-1.0f, row);
        }

        public Matrix colAdd(Matrix col) {
            return this.colComb(1.0f, col);
        }

        public Matrix colAddi(Matrix col) {
            return this.colCombi(1.0f, col);
        }

        public Matrix colSub(Matrix col) {
            return this.colComb(-1.0f, col);
        }

        public Matrix colSubi(Matrix col) {
            return this.colCombi(-1.0f, col);
        }

        public Matrix rowSum() {
            Matrix ones = Matrix.ones(1, this.rows);
            return this.dot(ones);
        }

        public Matrix colSum() {
            Matrix ones = Matrix.ones(this.cols, 1);
            return ones.dot(this);
        }

        public Matrix sub(Matrix that) {
            return comb(1.0f, -1.0f, that);
        }

        public Matrix subi(Matrix that) {
            replaceRef(sub(that), this);
            return this;
        }

        public Matrix add(Matrix that) {
            Matrix result = comb(1.0f, 1.0f, that);
            return comb(1.0f, 1.0f, that);
        }

        public Matrix addi(Matrix that) {
            replaceRef(add(that), this);
            return this;
        }

        public Matrix rowComb(float alpha, Matrix row) {
            Matrix result = this.copy();
            return result.rowCombi(alpha, row);
        }

        public Matrix rowCombi(float alpha, Matrix row) {
            Matrix weights = Matrix.ones(this.rows, 1);
            ger(alpha, weights, row, this);
            return this;
        }

        public Matrix colComb(float alpha, Matrix row) {
            Matrix result = this.copy();
            return result.colCombi(alpha, row);
        }

        public Matrix colCombi(float alpha, Matrix col) {
            Matrix weights = Matrix.ones(this.cols, 1);
            ger(alpha, col, weights, this);
            return this;
        }

        // result = alpha * this + beta * that
        public Matrix comb(float alpha, float beta, Matrix that) {
            Matrix result = new Matrix(rows, cols);
            JCublas2.cublasSgeam(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, rows, cols, Pointer.to(new float[]{alpha}), data_d, rows, Pointer.to(new float[]{beta}), that.data_d, that.rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix combi(float alpha, float beta, Matrix that) {
            replaceRef(comb(alpha, beta, that), this);
            return this;
        }

        public Matrix dot(Matrix that) {
            Matrix result = new Matrix(this.rows, that.cols);
            //dot(this, that, result);
            gemm(1.0f, this, that, 0.0f, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix dotT(Matrix that) {
            Matrix result = new Matrix(this.rows, that.rows);
            CublasUtil.Matrix transpose = that.transpose();
            gemm(1.0f, this, transpose, 0.0f, result);
            transpose.free();
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        /*public Matrix mmuli(Matrix that) {
            replaceRef(mmul(that), this);
            return this;
        }*/

        public Matrix add(float alpha) {
            Matrix result = new Matrix(rows, cols);
            scalarAdd(this, alpha, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix addi(float alpha) {
            replaceRef(add(alpha), this);
            return this;
        }

        public Matrix log() {
            Matrix result = new Matrix(rows, cols);
            log(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix logi() {
            replaceRef(log(), this);
            return this;
        }

        public Matrix exp() {
            Matrix result = new Matrix(rows, cols);
            exp(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix expi() {
            replaceRef(exp(), this);
            return this;
        }

        public Matrix sign() {
            Matrix result = new Matrix(rows, cols);
            sign(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix signi() {
            replaceRef(sign(), this);
            return this;
        }

        public Matrix abs() {
            Matrix result = new Matrix(rows, cols);
            abs(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix absi() {
            replaceRef(abs(), this);
            return this;
        }

        public Matrix mul(Matrix that) {
            Matrix result = new Matrix(rows, cols);
            mul(this, that, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix muli(Matrix that) {
            replaceRef(mul(that), this);
            return this;
        }

        public Matrix mul(float alpha) {
            return copy().muli(alpha);
        }

        public Matrix muli(float alpha) {
            JCublas2.cublasSscal(cublasHandle, rows * cols, Pointer.to(new float[]{alpha}), data_d, 1);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public Matrix div(Matrix that) {
            Matrix result = new Matrix(rows, cols);
            div(this, that, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public void mask(float val, float newVal, Matrix that) {
            mask(that, val, newVal, this);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public Matrix Softmax(Matrix that) {
            Matrix result = new Matrix(rows, cols);
            softmax_sum(that, this.rows, this.cols, result);
            softmax_probability(result, this.rows, this.cols, this);
            result.free();
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public void addCopy(Matrix that, int start) {
            addCopy(that, this, this.rows, this.cols, start);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public void add_(Matrix that) {
            Add(this, that);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public void addBackCopy(Matrix that, int start) {
            addBackCopy(that, this, this.rows, this.cols, start);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public void div(float that) {
            div(this, that);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public Matrix divi(Matrix that) {
            replaceRef(div(that), this);
            return this;
        }

        public Matrix max(float alpha) {
            Matrix result = new Matrix(rows, cols);
            max(this, result, alpha);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix maxi(float alpha) {
            replaceRef(max(alpha), this);
            return this;
        }

        public Matrix min(float alpha) {
            Matrix result = new Matrix(rows, cols);
            min(this, result, alpha);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix mini(float alpha) {
            replaceRef(min(alpha), this);
            return this;
        }

        public Matrix pow(float alpha) {
            Matrix result = new Matrix(rows, cols);
            pow(this, result, alpha);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix powi(float alpha) {
            replaceRef(pow(alpha), this);
            return this;
        }

        public Matrix sqr() {
            Matrix result = new Matrix(rows, cols);
            sqr(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix sqri() {
            replaceRef(sqr(), this);
            return this;
        }

        public Matrix sqrt() {
            Matrix result = new Matrix(rows, cols);
            sqrt(this, result);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public void softmax(Matrix matrix) {
            Matrix auxE = new Matrix(cols, rows);
            auxE.set(0);
            softmax(matrix, auxE, cols, this);
            auxE.free();
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public void derSoftmax(Matrix matrix, Matrix error) {
            derSoftmax(matrix, error, this);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public void dropout(float random, float chanceDrop) {
            dropout(this, random, chanceDrop);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        public Matrix sqrti() {
            replaceRef(sqrt(), this);
            return this;
        }

        public Matrix transpose() {
            Matrix result = new Matrix(cols, rows);
            JCublas2.cublasSgeam(cublasHandle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T, cols, rows, Pointer.to(new float[]{1.0f}), data_d, rows, Pointer.to(new float[]{0.0f}), new Pointer(), rows, result.data_d, result.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result;
        }

        public Matrix transposei() {
            if (isScalar()) {
            } else if (isVector()) {
                int rowsTmp = this.rows;
                this.rows = this.cols;
                this.cols = rowsTmp;
            } else {
                replaceRef(transpose(), this);
            }
            return this;
        }

        public Matrix zeroi() {
            JCublas2.cublasSgeam(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, rows, cols, Pointer.to(new float[]{0.0f}), new Pointer(), rows, Pointer.to(new float[]{0.0f}), new Pointer(), rows, data_d, rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return this;
        }

        public float norm1() {
            float[] result = new float[1];
            JCublas2.cublasSasum(cublasHandle, rows * cols, data_d, 1, Pointer.to(result));
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result[0];
        }

        public float norm2() {
            float[] result = new float[1];
            JCublas2.cublasSnrm2(cublasHandle, rows * cols, data_d, 1, Pointer.to(result));
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            return result[0];
        }

        public float distance1(Matrix that) {
            return comb(1.0f, -1.0f, that).norm1();
        }

        public float distance2(Matrix that) {
            return comb(1.0f, -1.0f, that).norm2();
        }

        //////////////////////////////////////////

        public static List<Matrix> invert(List<Matrix> A) {
            List<Matrix> B = new ArrayList<Matrix>();
            for (int i = 0; i < A.size(); ++i) {
                B.add(new Matrix(A.get(0).rows, A.get(0).cols));
            }
            getrfGetriBatched(A, B);
            return B;
        }

        public static List<Matrix> mmul(List<Matrix> A, List<Matrix> B) {
            List<Matrix> C = new ArrayList<Matrix>();
            for (int i = 0; i < A.size(); ++i) {
                C.add(new Matrix(A.get(0).rows, B.get(0).cols));
            }
            gemmBatched(1.0f, A, B, 0.0f, C);
            return C;
        }

        //////////////////////////////////////////

        private static float[] toColMajor(float[][] mat) {
            int rows = mat.length;
            int cols = mat[0].length;
            float[] data = new float[rows * cols];
            int i = 0;
            for (int c = 0; c < cols; ++c) {
                for (int r = 0; r < rows; ++r) {
                    data[i] = mat[r][c];
                    i++;
                }
            }
            return data;
        }

        private static float[][] fromColMajor(float[] data, int rows) {
            int cols = data.length / rows;
            float[][] mat = new float[rows][cols];
            int i = 0;
            for (int c = 0; c < cols; ++c) {
                for (int r = 0; r < rows; ++r) {
                    mat[r][c] = data[i];
                    i++;
                }
            }
            return mat;
        }

        private static void replaceRef(Matrix A, Matrix B) {
            B.free();
            B.rows = A.rows;
            B.cols = A.cols;
            B.data_d = A.data_d;
        }

        private static final int BLOCK_SIZE = 1024;

        // batched inverse
        private static void getrfGetriBatched(List<Matrix> A, List<Matrix> B) {
            Pointer[] Apointers = new Pointer[A.size()];
            Pointer[] Bpointers = new Pointer[B.size()];
            for (int i = 0; i < A.size(); ++i) {
                Apointers[i] = A.get(i).data_d;
                Bpointers[i] = B.get(i).data_d;
            }
            Pointer Apointers_d = new Pointer();
            cudaMalloc(Apointers_d, A.size() * Sizeof.POINTER);
            cudaMemcpy(Apointers_d, Pointer.to(Apointers), A.size() * Sizeof.POINTER, cudaMemcpyHostToDevice);
            Pointer Bpointers_d = new Pointer();
            cudaMalloc(Bpointers_d, B.size() * Sizeof.POINTER);
            cudaMemcpy(Bpointers_d, Pointer.to(Bpointers), B.size() * Sizeof.POINTER, cudaMemcpyHostToDevice);
            Pointer info_d = new Pointer();
            cudaMalloc(info_d, A.size() * Sizeof.INT);
            Pointer pivots_d = new Pointer();
            cudaMalloc(pivots_d, A.get(0).rows * A.size() * Sizeof.INT);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCublas2.cublasSgetrfBatched(cublasHandle, A.get(0).rows, Apointers_d, A.get(0).rows, pivots_d, info_d, A.size());
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCublas2.cublasSgetriBatched(cublasHandle, A.get(0).rows, Apointers_d, A.get(0).rows, pivots_d, Bpointers_d, B.get(0).rows, info_d, A.size());
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(Apointers_d);
            JCuda.cudaFree(Bpointers_d);
            JCuda.cudaFree(info_d);
            JCuda.cudaFree(pivots_d);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // batched C = alpha * A * B + beta * C
        private static void gemmBatched(float alpha, List<Matrix> A, List<Matrix> B, float beta, List<Matrix> C) {
            Pointer[] Apointers = new Pointer[A.size()];
            Pointer[] Bpointers = new Pointer[B.size()];
            Pointer[] Cpointers = new Pointer[C.size()];
            for (int i = 0; i < A.size(); ++i) {
                Apointers[i] = A.get(i).data_d;
                Bpointers[i] = B.get(i).data_d;
                Cpointers[i] = C.get(i).data_d;
            }
            Pointer Apointers_d = new Pointer();
            cudaMalloc(Apointers_d, A.size() * Sizeof.POINTER);
            cudaMemcpy(Apointers_d, Pointer.to(Apointers), A.size() * Sizeof.POINTER, cudaMemcpyHostToDevice);
            Pointer Bpointers_d = new Pointer();
            cudaMalloc(Bpointers_d, B.size() * Sizeof.POINTER);
            cudaMemcpy(Bpointers_d, Pointer.to(Bpointers), B.size() * Sizeof.POINTER, cudaMemcpyHostToDevice);
            Pointer Cpointers_d = new Pointer();
            cudaMalloc(Cpointers_d, C.size() * Sizeof.POINTER);
            cudaMemcpy(Cpointers_d, Pointer.to(Cpointers), C.size() * Sizeof.POINTER, cudaMemcpyHostToDevice);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCublas2.cublasSgemmBatched(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, C.get(0).rows, C.get(0).cols, B.get(0).rows, Pointer.to(new float[]{alpha}), Apointers_d, A.get(0).rows, Bpointers_d, B.get(0).rows, Pointer.to(new float[]{beta}), Cpointers_d, C.get(0).rows, A.size());
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            JCuda.cudaFree(Apointers_d);
            JCuda.cudaFree(Bpointers_d);
            JCuda.cudaFree(Cpointers_d);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // C = alpha * A * B + beta * C
        private static void gemm(float alpha, Matrix A, Matrix B, float beta, Matrix C) {
            //JCublas2.cublasSgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, C.rows, C.cols, B.rows, Pointer.to(new float[] {alpha}), A.data_d, A.rows, B.data_d, B.rows, Pointer.to(new float[] {beta}), C.data_d, C.rows);
            JCublas2.cublasSgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, B.cols, A.rows, B.rows, Pointer.to(new float[]{alpha}), B.data_d, B.cols, A.data_d, B.rows, Pointer.to(new float[]{beta}), C.data_d, B.cols);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = A * diag(x) or B = diag(x) * A
        private static void dgmm(Matrix A, Matrix x, Matrix B, boolean left) {
            JCublas2.cublasSdgmm(cublasHandle, left ? cublasSideMode.CUBLAS_SIDE_LEFT : cublasSideMode.CUBLAS_SIDE_RIGHT, A.rows, A.cols, A.data_d, A.rows, x.data_d, 1, B.data_d, B.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // A = alpha * x * y^T + A
        private static void ger(float alpha, Matrix x, Matrix y, Matrix A) {
            JCublas2.cublasSger(cublasHandle, A.rows, A.cols, Pointer.to(new float[]{alpha}), x.data_d, 1, y.data_d, 1, A.data_d, A.rows);
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // A = alpha
        private static void scalarSet(Matrix A, float alpha) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorScalarSet");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(new float[]{alpha}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = A + alpha
        private static void scalarAdd(Matrix A, float alpha, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorScalarAdd");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new float[]{alpha}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = log(A)
        private static void log(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorLog");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = exp(A)
        private static void exp(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorExp");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = sign(A)
        private static void sign(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorSign");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = abs(A)
        private static void abs(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorAbs");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // C = A ./ B
        private static void div(Matrix A, Matrix B, Matrix C) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorDiv");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(C.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void matrixMultiplyShared(Matrix A, Matrix B, Matrix C) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "matrixMultiplyShared");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(C.data_d), Pointer.to(new float[]{A.rows}), Pointer.to(new int[]{A.cols}), Pointer.to(new int[]{B.rows}), Pointer.to(new int[]{B.cols}), Pointer.to(new int[]{C.rows}), Pointer.to(new int[]{C.cols}));
            cuLaunchKernel(function,
                    (C.cols - 1) / 32 + 1, (C.rows - 1) / 32 + 1, 1,      // Grid dimension
                    32, 32, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // A = A ./ B
        private static void div(Matrix A, float B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "matrixDiv");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(new float[]{B}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void dot(Matrix A, Matrix B, Matrix C) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "dot");
            int blockSize = 32;
            int grid_rows = (int) Math.ceil((double) C.cols / blockSize);
            int grid_cols = (int) Math.ceil((double) C.rows / blockSize);

            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(C.data_d), Pointer.to(new int[]{A.cols}), Pointer.to(new int[]{C.rows}), Pointer.to(new int[]{C.cols}));
            cuLaunchKernel(function,
                    grid_rows, grid_cols, 1,      // Grid dimension
                    blockSize, blockSize, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void mask(Matrix A, float val, float newVal, Matrix C) {
            int n = C.rows * C.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "mask");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(new float[]{val}), Pointer.to(new float[]{newVal}), Pointer.to(C.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void softmax_sum(Matrix A, int row, int col, Matrix sum) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "softmax_sum");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(new int[]{row}), Pointer.to(new int[]{col}), Pointer.to(sum.data_d));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void softmax_probability(Matrix sum, int row, int col, Matrix C) {
            int n = row * col;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "softmax_probability");
            Pointer kernelParameters = Pointer.to(Pointer.to(sum.data_d), Pointer.to(new int[]{row}), Pointer.to(new int[]{col}), Pointer.to(C.data_d));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void addCopy(Matrix A, Matrix C, int row, int col, int start) {
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "addCopy");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(C.data_d), Pointer.to(new int[]{A.cols()}), Pointer.to(new int[]{C.cols()}), Pointer.to(new int[]{start}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void Add(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "MatAdd");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void addBackCopy(Matrix A, Matrix C, int row, int col, int start) {
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "addBackCopy");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(C.data_d), Pointer.to(new int[]{A.cols()}), Pointer.to(new int[]{C.cols()}), Pointer.to(new int[]{start}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // C = A .* B
        private static void mul(Matrix A, Matrix B, Matrix C) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorMul");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(C.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void dropout(Matrix A, float random, float chanceDrop) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "dropout");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(new float[]{random}), Pointer.to(new float[]{chanceDrop}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = max(A, val)
        private static void max(Matrix A, Matrix B, float val) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorMax");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = min(A, val)
        private static void min(Matrix A, Matrix B, float val) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorMin");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = pow(A, val)
        private static void pow(Matrix A, Matrix B, float val) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorPow");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new float[]{val}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = sqr(A)
        private static void sqr(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorSqr");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        // B = sqrt(A)
        private static void sqrt(Matrix A, Matrix B) {
            int n = A.rows * A.cols;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "vectorSqrt");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(B.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void softmax(Matrix A, Matrix auxE, int sample_dim, Matrix C) {
            int n = A.rows;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "Softmax");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(auxE.data_d), Pointer.to(new int[]{sample_dim}), Pointer.to(C.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static void derSoftmax(Matrix A, Matrix auxE, Matrix C) {
            int n = A.rows;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "derSoftmax");
            Pointer kernelParameters = Pointer.to(Pointer.to(A.data_d), Pointer.to(auxE.data_d), Pointer.to(new int[]{A.cols}), Pointer.to(C.data_d), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }

        private static int Dim(int L_row, int L_col, int R_row, int R_col) {
            int sqr_dim_X, sqr_dim_Y, size;

            sqr_dim_X = R_row;
            if (L_row > R_row) {
                sqr_dim_X = L_row;
            }

            sqr_dim_Y = R_col;
            if (L_col > R_col) {
                sqr_dim_Y = L_col;
            }

            size = sqr_dim_Y;
            if (sqr_dim_X > sqr_dim_Y) {
                size = sqr_dim_X;
            }

            int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
            size = temp * BLOCK_SIZE;
            return size;
        }

        public static final String kernels =
                "extern \"C\"\n" +
                        "__global__ void vectorScalarSet(float* A, float alpha, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        A[i] = alpha;\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void matrixDiv(float* A, float B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        A[i] /= B;\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorScalarAdd(const float* __restrict__ A, float* B, float alpha, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = A[i] + alpha;\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorLog(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = log(A[i]);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorExp(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = exp(A[i]);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorSign(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = (A[i] > 0.0 ? 1.0 : -1.0);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorAbs(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = abs(A[i]);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorDiv(const float* __restrict__ A, const float* __restrict__ B, float* C, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        C[i] = A[i] / B[i];\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorMul(const float* __restrict__ A, const float* __restrict__ B, float* C, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        C[i] = A[i] * B[i];\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorMax(const float* __restrict__ A, float* B, float val, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = max(A[i], val);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorMin(const float* __restrict__ A, float* B, float val, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = min(A[i], val);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorPow(const float* __restrict__ A, float* B, float val, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = pow((double) A[i], (double) val);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorSqr(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    float val;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        val = A[i];\n" +
                        "        B[i] = val*val;\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void vectorSqrt(const float* __restrict__ A, float* B, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "        B[i] = sqrt(A[i]);\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void mask(const float* __restrict__ A, float val, float newVal, float* C, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "       if (A[i] == val)\n" +
                        "       {\n" +
                        "           C[i] = newVal;\n" +
                        "       }\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void dropout(float* A, float random, float chanceDrop, int numElements)\n" +
                        "{\n" +
                        "    int i = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    float drop = 1.0f / (1.0f - chanceDrop);\n" +
                        "    if (i < numElements)\n" +
                        "    {\n" +
                        "       if (random > chanceDrop)\n" +
                        "       {\n" +
                        "           A[i] = A[i] * drop;\n" +
                        "       }\n" +
                        "    }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error,int column, float* C, int numElements)\n" +
                        "{\n" +
                        "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (k < numElements)\n" +
                        "    {\n" +
                        "       int index, indexI, indexJ;\n" +
                        "       float value;\n" +
                        "       index = k * column;\n" +
                        "       indexI = index;\n" +
                        "       for (int i = 0; i < column; i++, indexI++)\n" +
                        "       {\n" +
                        "           C[indexI] = 0;\n" +
                        "           indexJ = index;\n" +
                        "           for (int j = 0; j < column; j++, indexJ++) \n" +
                        "           {\n" +
                        "               if (i != j) \n" +
                        "               {\n" +
                        "                   value = output[indexI] * -output[indexJ];\n" +
                        "               } \n" +
                        "               else \n" +
                        "               {\n" +
                        "                   value = output[indexI] * (1 - output[indexI]);\n" +
                        "               }\n" +
                        "               C[indexI] += error[indexJ] * value;\n" +
                        "           }\n" +
                        "       }\n" +
                        "   }\n" +
                        "}\n" +

                        "extern \"C\"\n" +
                        "__global__ void MatAdd(float* A, const float* __restrict__ B, int numElements)\n" +
                        "{\n" +
                        "    int k = blockDim.x * blockIdx.x + threadIdx.x;\n" +
                        "    if (k < numElements)\n" +
                        "    {\n" +
                        "       A[k] += B[k];\n" +
                        "    }\n" +
                        "}\n"+

                        "extern \"C\"\n"+
                        "__global__ void addCopy(float* A, float* C, int A_col, int C_col, int start, int n) \n"+
                        "{\n"+
                            "int index = threadIdx.x + (blockIdx.x * blockDim.x);\n"+
                            "if (index >= n)\n"+
                            "   return;\n"+
                            "int indexOut = 0;\n"+
                            "int indexIn = index * C_col + start * A_col;\n"+
                            "for (int j = 0; j < A_col; j++, indexIn++, indexOut++) \n"+
                            "{\n"+
                            "    C[indexIn] = A[indexOut];\n"+
                            "}\n"+
                        "}\n"+

                        "extern \"C\"\n"+
                        "__global__ void addBackCopy(float* A, float* C, int A_col, int C_col, int start, int n) \n"+
                        "{\n"+
                        "   int index = threadIdx.x + (blockIdx.x * blockDim.x);\n"+
                        "   if (index >= n)\n"+
                        "       return;\n"+
                        "   int indexOut = 0;\n"+
                        "   int indexIn = index * A_col + start * C_col;\n"+
                        "   for (int j = 0; j < C_col; j++, indexIn++, indexOut++) \n"+
                        "   {\n"+
                        "       C[indexIn] = A[indexOut];\n"+
                        "   }\n"+
                        "}\n"+

                        "extern \"C\"\n"+
                        "__global__ void Softmax(const float* __restrict__ A, float* auxE, int sample_dim, float* N, int numElements)\n"+
                        "{\n"+
                            //This way of programing allow no warp syncronization as we only need kernel optimization.
                            //Maybe use thrust library but could be tricky. Should study if it fit well with this problem. Think is not, we would need one thrust vector per row.
                            //On the other hand possibly implement reduction as in http://www.cuvilib.com/Reduction.pdf. Will need to call another function. This could be complecated also as we need to see which thread id implements softmax and which one computes maximum. For now simple approximation.
                            "float C_value = 0;\n"+
                            "int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;\n"+
                            "float maxCoef = A[thread_id_x*sample_dim];\n"+
                            "float actualCoef = 0;\n"+
                            "double E = 2.718281828459045;\n"+
                            "if (thread_id_x < numElements)\n"+
                            "{\n"+
                                ///REALLY HIGH PROBABILITY OF BRANCH DIVERGENCE.
                                //Description: All of the threads that lie under one condition execute first (stalling the others) and then next. Assuming one clock cycle per operation we would need double time to execute one warp.
                                //Warping divergence: study reduction options for getting the maximum
                                "#pragma omp parallel for\n"+
                                "for (int cA = 1; cA < sample_dim; cA++)\n"+
                                    "if (A[thread_id_x * sample_dim + cA] > maxCoef)\n"+
                                        "maxCoef = A[thread_id_x * sample_dim+cA];\n"+

                                //No warping divergence as all threads execute the same
                                "#pragma omp parallel for\n"+
                                "for (int cA = 0; cA < sample_dim; cA++)\n"+
                                "{\n"+
                                    "actualCoef = (float) pow(E, (double)(A[thread_id_x * sample_dim + cA] - maxCoef));\n"+
                                    "auxE[thread_id_x * sample_dim + cA] = actualCoef;\n"+
                                    "C_value += actualCoef;\n"+
                                "}\n"+
                                "#pragma omp parallel for\n"+
                                "C_value += 0.00000001f;\n" +
                                "for (int cA = 0; cA < sample_dim; cA++)\n"+
                                "{\n"+
                                    "N[thread_id_x * sample_dim + cA] = auxE[thread_id_x * sample_dim + cA] / C_value;\n"+
                                "}\n"+
                            "}\n"+
                        "}\n"+
                        "\n"

                ;

    }

    /*public void MultiHeadAttn(jcuda.jcudnn.cudnnHandle Handle) {
        int batch_size 		=  16;
        int emb_dim 		=  1024;
	    int num_heads       =  16;
        int seq_len 		=  64;
        int beam_dim 		=  1;
        int seqLensVecSize = batch_size;
        List<Integer> seqLensVec = new ArrayList<>();
        for (int i = 0; i < seqLensVecSize; i++)
            seqLensVec.add(seq_len);

        // q, k, v embedding vector lengths
        int qSize = emb_dim;
        int kSize = emb_dim;
        int vSize = emb_dim;

        // q, k, v embedding vector lengths after input projections.
        int qProjSize = emb_dim / num_heads;
        int kProjSize = emb_dim / num_heads;
        int vProjSize = emb_dim / num_heads;

        // o embedding vector length after the output projection
        int oProjSize = emb_dim;

        int qoMaxSeqLength = seq_len;
        int kvMaxSeqLength = seq_len;
        int maxBatchSize = batch_size;
        int maxBeamSize = 1;

        // ***********************************************************************
        // Variables' Roles:
        //
        // devAQ -> [ devWQ + devBQ ] -> ..
        // Input      Linear Layer          \
        //                                   |
        // devAK -> [ devWK + devBK ] -> [ hidden ] -> [ devWO + devBO ] -> devAO
        // Input      Linear Layer           |           Linear Layer       Output
        //                                  /
        // devAV -> [ devWV + devBV ] -> ..
        // Input      Linear Layer
        //
        // ***********************************************************************

        // Below variables are used as shown above.
        //Pointer Apointers_d = new Pointer();
        //JCuda.cudaMalloc(Apointers_d, A.size() * Sizeof.POINTER);
        //JCuda.cudaMemcpy(Apointers_d, Pointer.to(Apointers), A.size() * Sizeof.POINTER, cudaMemcpyKind.cudaMemcpyHostToDevice);

        Pointer devAQ = new Pointer(); // q Activations
        Pointer devAK = new Pointer(); // k Activations
        Pointer devAV = new Pointer(); // v Activations
        Pointer devAO = new Pointer(); // o Activations
        Pointer devWQ = new Pointer(); // q Linear Layer Weights
        Pointer devWK = new Pointer(); // k Linear Layer Weights
        Pointer devWV = new Pointer(); // v Linear Layer Weights
        Pointer devWO = new Pointer(); // o Linear Layer Weights
        Pointer devBQ = new Pointer(); // q Linear Layer Biases
        Pointer devBK = new Pointer(); // k Linear Layer Biases
        Pointer devBV = new Pointer(); // v Linear Layer Biases
        Pointer devBO = new Pointer(); // o Linear Layer Biases

        // Corresponding partial derivatives.
        Pointer devDAQ = new Pointer();
        Pointer devDAK = new Pointer();
        Pointer devDAV = new Pointer();
        Pointer devDAO = new Pointer();
        Pointer devDWQ = new Pointer();
        Pointer devDWK = new Pointer();
        Pointer devDWV = new Pointer();
        Pointer devDWO = new Pointer();
        Pointer devDBQ = new Pointer();
        Pointer devDBK = new Pointer();
        Pointer devDBV = new Pointer();
        Pointer devDBO = new Pointer();

        long[] sizeWeights = new long[]{0};
        long[] sizeWkspace = new long[]{0};
        long[] sizeReserve = new long[]{0};
        Pointer devWs = new Pointer();
        Pointer devDWs = new Pointer();
        Pointer devWkspace = new Pointer();
        Pointer devReserve = new Pointer();

        // Device array specifying seq. lengths of query, residual, and output seq. data.
        Pointer devQSeqArray = new Pointer();
        // Device array specifying seq. lengths of key and value input data.
        Pointer devKSeqArray = new Pointer();
        // Host arrays specifying the attention window size for each Q time-step.
        List<Integer> loWinIdx, hiWinIdx;

        int CUDNN_SEQDATA_DIM_COUNT = 4;

        int axes[] = new int[CUDNN_SEQDATA_DIM_COUNT];
        axes[0] = jcuda.jcudnn.cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM;
        axes[1] = jcuda.jcudnn.cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM;
        axes[2] = jcuda.jcudnn.cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM;
        axes[3] = jcuda.jcudnn.cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM;

        int dimA[] = new int[CUDNN_SEQDATA_DIM_COUNT];
        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BEAM_DIM]  = beam_dim;
        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_TIME_DIM]  = seq_len;
        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_BATCH_DIM] = batch_size;
        dimA[cudnnSeqDataAxis.CUDNN_SEQDATA_VECT_DIM]  = emb_dim;

        jcuda.jcudnn.cudnnAttnDescriptor attn_desc;
        jcuda.jcudnn.cudnnSeqDataDescriptor q_desc = null;
        jcuda.jcudnn.cudnnSeqDataDescriptor k_desc = null;
        jcuda.jcudnn.cudnnSeqDataDescriptor v_desc = null;
        jcuda.jcudnn.cudnnSeqDataDescriptor o_desc = null;

        // Dropout is currently not supported by CuDNN's multi-head attention API.
        jcuda.jcudnn.cudnnDropoutDescriptor attnDropoutDesc = new jcuda.jcudnn.cudnnDropoutDescriptor();
        jcuda.jcudnn.cudnnDropoutDescriptor postDropoutDesc = new jcuda.jcudnn.cudnnDropoutDescriptor();

        boolean enable_bias = false;
        int attnMode = enable_bias ? JCudnn.CUDNN_ATTN_ENABLE_PROJ_BIASES : JCudnn.CUDNN_ATTN_DISABLE_PROJ_BIASES;
        double smScaler = 1.0;
        int dataType = cudnnDataType.CUDNN_DATA_FLOAT;
        int computePrec = cudnnDataType.CUDNN_DATA_FLOAT;
        int mathType = cudnnMathType.CUDNN_DEFAULT_MATH;

        checkCudnnError(jcuda.jcudnn.JCudnn.cudnnCreateAttnDescriptor(attn_desc));
        checkCudnnError(jcuda.jcudnn.JCudnn.cudnnSetAttnDescriptor(
                attn_desc,
                attnMode,
                num_heads,
                smScaler,
                dataType,
                computePrec,
                mathType,
                attnDropoutDesc,
                postDropoutDesc,
                qSize,
                kSize,
                vSize,
                qProjSize,
                kProjSize,
                vProjSize,
                oProjSize,
                qoMaxSeqLength,
                kvMaxSeqLength,
                maxBatchSize,
                maxBeamSize));

        boolean training = true;

        checkCudnnError(jcuda.jcudnn.JCudnn.cudnnCreate(Handle));
        checkCudnnError(jcuda.jcudnn.JCudnn.cudnnGetMultiHeadAttnBuffers(
                Handle, attn_desc, sizeWeights, sizeWkspace, training ? sizeReserve : null));
        checkCudaErrors(cudaMalloc(devWs, sizeWeights));
        checkCudaErrors(cudaMalloc(devWkspace, sizeWkspace));
        if (training) {
            checkCudaErrors(cudaMalloc(devDWs, sizeWeights));
            checkCudaErrors(cudaMalloc(devReserve, sizeReserve));
            checkCudaErrors(JCuda.cudaMemset(devDWs, 0, sizeWeights));
            checkCudaErrors(JCuda.cudaMemset(devReserve, 0, sizeReserve));

            checkCudaErrors(JCuda.cudaMemset(devWs, 0, sizeWeights));
            checkCudaErrors(JCuda.cudaMemset(devWkspace, 0, sizeWkspace));
        }

        class subblock{
            String name;
            Pointer devA; 	// used for q, k, v, o activations.
            Pointer devW;  	// used for q, k, v, o linear layer weights.
            Pointer devB;  	// used for q, k, v, o linear layer biases.
            Pointer devDA; 	// used for partial derivatives of q, k, v, o activations.
            Pointer devDW;  // used for partial derivatives of q, k, v, o linear layer weights.
            Pointer devDB;  // used for partial derivatives of q, k, v, o linear layer biases.
            cudnnMultiHeadAttnWeightKind enumW;
            cudnnMultiHeadAttnWeightKind enumB;
        };

        // Shorten enum names to fit the subblocks table below.
        int enumWQ = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_Q_WEIGHTS, enumBQ = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_Q_BIASES;
        int enumWK = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_K_WEIGHTS, enumBK = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_K_BIASES;
        int enumWV = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_V_WEIGHTS, enumBV = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_V_BIASES;
        int enumWO = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_O_WEIGHTS, enumBO = jcuda.jcudnn.cudnnMultiHeadAttnWeightKind.CUDNN_MH_ATTN_O_BIASES;

        List<subblock> subblocks = new ArrayList<subblock>()
        {
            //
            //  Corresponding struct member names:
            // .name | .devA | .devW | .devB | .devDA | .devDW | .devDB |.enumW |.enumB
            //
            {"q",     devAQ, devWQ, devBQ, devDAQ, devDWQ, devDBQ, enumWQ, enumBQ},
            {"k",     devAK, devWK, devBK, devDAK, devDWK, devDBK, enumWK, enumBK},
            {"v",     devAV, devWV, devBV, devDAV, devDWV, devDBV, enumWV, enumBV},
            {"o",     devAO, devWO, devBO, devDAO, devDWO, devDBO, enumWO, enumBO},
        };
        for (subblock s : subblocks) {

            auto avec = cfg.vecs[s.name];
            auto wvec = cfg.vecs[s.name + "_p.weight"];
            auto bvec = cfg.vecs[s.name + "_p.bias"];

            cudnnTensorDescriptor desc;
            checkCudnnError(cudnnCreateTensorDescriptor(desc));

            // Allocate memory for activations devAQ, devAK, devAV, devAO.
            checkCudaErrors(cudaMalloc(s.devA, sizeof(T) * avec.size()));

            // Store addresses for weights in devWQ, devWK, devWV, devWO.
            checkCudnnError(cudnnGetMultiHeadAttnWeights(
                    cudnnHandle, attn_desc, s.enumW, sizeWeights, devWs,  desc, s.devW));

            // Store addresses for biases in devBQ, devBK, devBV, devBO.
            checkCudnnError(cudnnGetMultiHeadAttnWeights(
                    cudnnHandle, attn_desc, s.enumB, sizeWeights, devWs,  desc, s.devB));

            if (training) {
                // Allocate memory for activations' gradients devDAQ, devDAK, devDAV, devDAO.
                checkCudaErrors(cudaMalloc(s.devDA, sizeof(T) * avec.size()));

                // Store addresses for weights' gradients in devDWQ, devDWK, devDWV, devDWO.
                checkCudnnError(cudnnGetMultiHeadAttnWeights(
                        cudnnHandle, attn_desc, s.enumW, sizeWeights, devDWs, desc, s.devDW));

                // Store addresses for biases' gradients in devDBQ, devDBK, devDBV, devDBO.
                checkCudnnError(cudnnGetMultiHeadAttnWeights(
                        cudnnHandle, attn_desc, s.enumB, sizeWeights, devDWs, desc, s.devDB));
            }
            // Copy PyTorch reference weights to GPU.
            checkCudaErrors(cudaMemcpy(
                    *s.devW, wvec.data(), sizeof(T) * wvec.size(), cudaMemcpyHostToDevice));

            // Copy PyTorch reference biases to GPU.
            checkCudaErrors(cudaMemcpy(
                    *s.devB, bvec.data(), sizeof(T) * bvec.size(), cudaMemcpyHostToDevice));

            // Copy PyTorch reference inputs to GPU.
            if (s.name == "q" || s.name == "k" || s.name == "v")
                checkCudaErrors(cudaMemcpy(
                        *s.devA, avec.data(), sizeof(T) * avec.size(), cudaMemcpyHostToDevice));
        }
        if (training) {
            // Copy gradients that will propagate backward through the entire net, to GPU.
            std::vector<T> DAO (cfg.vecs["o"].size(), 1.0);
            checkCudaErrors(cudaMemcpy(
                    devDAO, DAO.data(), sizeof(T) * DAO.size(), cudaMemcpyHostToDevice));
        }
        for (cudnnSeqDataDescriptor * desc: { &q_desc, &k_desc, &v_desc, &o_desc }) {
            checkCudnnError(cudnnCreateSeqDataDescriptor(desc));
            checkCudnnError(cudnnSetSeqDataDescriptor(
                    *desc, dataType, CUDNN_SEQDATA_DIM_COUNT, dimA, axes, seqLensVecSize, seqLensVec.data(), NULL));
        }

        std::vector<int>  hostQSeqVec(batch_size * beam_dim, seq_len);
        std::vector<int>  hostKSeqVec(batch_size, seq_len);
        int qSeqArraySize = hostQSeqVec.size() * sizeof(int);
        int kSeqArraySize = hostKSeqVec.size() * sizeof(int);

        checkCudaErrors(cudaMalloc((void**)&devQSeqArray, qSeqArraySize));
        checkCudaErrors(cudaMalloc((void**)&devKSeqArray, kSeqArraySize));
        checkCudaErrors(cudaMemcpy(
                devQSeqArray, (void*)hostQSeqVec.data(), qSeqArraySize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(
                devKSeqArray, (void*)hostKSeqVec.data(), kSeqArraySize, cudaMemcpyHostToDevice));

        int maxSeqLenK = INT_MAX;
        for (int i = 0; i < seq_len; i++) {
            loWinIdx.push_back(0);
            hiWinIdx.push_back(maxSeqLenK);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < FLAGS_iterations; ++iter) {
            int currIdx=-1;
            checkCudnnError(cudnnMultiHeadAttnForward(
                    // parameter names in CuDNN API docs
                    cudnnHandle,    // cudnnHandle_t handle
                    attn_desc,      // const cudnnAttnDescriptor_t attn_desc
                    currIdx,        // int currIdx
                    (const int*)loWinIdx.data(),    // const int loWinIdx[]
                    (const int*)hiWinIdx.data(),    // const int hiWinIdx[]
                    devQSeqArray,   // const int devQSeqArray[],
                    devKSeqArray,   // const int devKSeqArray[],
                    q_desc,         // const cudnnSeqDataDescriptor_t q_desc
                    devAQ,          // const void *queries,
                    NULL,           // const void *residuals,
                    k_desc,         // const cudnnSeqDataDescriptor_t k_desc,
                    devAK,          // const void *keys,
                    v_desc,         // const cudnnSeqDataDescriptor_t v_desc,
                    devAV,          // const void *values,
                    o_desc,         // const cudnnSeqDataDescriptor_t o_desc,
                    devAO,          // void *out
                    sizeWeights,    // size_t weightSizeInBytes,
                    devWs,          // const void *weights,
                    sizeWkspace,    // size_t workSpaceSizeInBytes,
                    devWkspace,     // void *workSpace,
                    sizeReserve,    // size_t reserveSpaceSizeInBytes,
                    devReserve));   // void *reserveSpace
            if (training) {
                checkCudnnError(cudnnMultiHeadAttnBackwardData(
                        cudnnHandle,
                        attn_desc,
                        (const int*)loWinIdx.data(),
                        (const int*)hiWinIdx.data(),
                        devQSeqArray,
                        devKSeqArray,
                        o_desc,
                        devDAO,
                        q_desc,
                        devDAQ,
                        devAQ,
                        k_desc,
                        devDAK,
                        devAK,
                        v_desc,
                        devDAV,
                        devAV,
                        sizeWeights,
                        devWs,
                        sizeWkspace,
                        devWkspace,
                        sizeReserve,
                        devReserve));
                checkCudnnError(cudnnMultiHeadAttnBackwardWeights(
                        cudnnHandle,
                        attn_desc,
                        CUDNN_WGRAD_MODE_SET,
                        q_desc,
                        devAQ,
                        k_desc,
                        devAK,
                        v_desc,
                        devAV,
                        o_desc,
                        devDAO,
                        sizeWeights,
                        devWs,
                        devDWs,
                        sizeWkspace,
                        devWkspace,
                        sizeReserve,
                        devReserve));
            }
        }

        checkCudaErrors(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();

        struct kv{
            std::string name;
            void * devPtr;
            T error = 0.0;
            int error_i = 0;
            T error_div_spread = 0.0;
            T minPyt = FLT_MAX;
            T maxPyt = FLT_MIN;
            T minCud = FLT_MAX;
            T maxCud = FLT_MIN;
        };
        std::vector<kv> kvvec {
            {"q_p.weight.grad", devDWQ},
            {"k_p.weight.grad", devDWK},
            {"v_p.weight.grad", devDWV},
            {"o_p.weight.grad", devDWO},
            // {"q_p.bias.grad", devDBQ},
            // {"k_p.bias.grad", devDBK},
            // {"v_p.bias.grad", devDBV},
            // {"o_p.bias.grad", devDBO},
            // {"q_p.weight", devWQ},
            // {"k_p.weight", devWK},
            // {"v_p.weight", devWV},
            // {"o_p.weight", devWO},
            // {"q_p.bias", devBQ},
            // {"k_p.bias", devBK},
            // {"v_p.bias", devBV},
            // {"o_p.bias", devBO},
            {"o", devAO},
        };

        for (auto & [name, devPtr, error, error_i, error_div_spread, minPyt, maxPyt, minCud, maxCud] : kvvec) {
            if (cfg.vecs.find(name) == cfg.vecs.end() || cfg.vecs[name].empty() || devPtr == nullptr)
                continue;
            auto pytVec = cfg.vecs[name];
            auto size = sizeof(T) * pytVec.size();
            auto cudnnVec = pytVec;
            checkCudaErrors(cudaMemcpy(cudnnVec.data(), devPtr, size, cudaMemcpyDeviceToHost));

            // TO DO (Maybe?): change indexing to using stride and dimension lengths
            // info provided by cudnnGetTensorNdDescriptor and/or cudnnGetSeqDataDescriptor.

            minPyt = sizeof(T) == 8 ? DBL_MAX : FLT_MAX;
            maxPyt = sizeof(T) == 8 ? DBL_MIN : FLT_MIN;
            minCud = sizeof(T) == 8 ? DBL_MAX : FLT_MAX;
            maxCud = sizeof(T) == 8 ? DBL_MIN : FLT_MIN;

            for (int i = 0; i < pytVec.size(); i++) {
                auto cudVal = cudnnVec[i];
                auto pytVal = pytVec[i];

                if (error < abs(cudVal - pytVal)) {
                    error = abs(cudVal - pytVal);
                    error_i = i;
                }
                minPyt = std::min(minPyt, pytVal);
                maxPyt = std::max(maxPyt, pytVal);
                minCud = std::min(minCud, cudVal);
                maxCud = std::max(maxCud, cudVal);
            }
        }

        JCuda.cudaFree(devWs);
        JCuda.cudaFree(devDWs);
        JCuda.cudaFree(devWkspace);
        JCuda.cudaFree(devReserve);
        JCuda.cudaFree(devQSeqArray);
        JCuda.cudaFree(devKSeqArray);
        JCuda.cudaFree(devAQ);
        JCuda.cudaFree(devAK);
        JCuda.cudaFree(devAV);
        JCuda.cudaFree(devAO);
        JCuda.cudaFree(devDAQ);
        JCuda.cudaFree(devDAK);
        JCuda.cudaFree(devDAV);
        JCuda.cudaFree(devDAO);

        return;
    }*/

    public void checkCudnnError(int status) {
        do {
            if (status != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
                FatalError("CUDNN failure: " + jcuda.jcudnn.JCudnn.cudnnGetErrorString(status));
            }
        } while (false);
    }

    public void checkCudaErrors(int status) {
        do {
            if (status != 0) {
                FatalError("Cuda failure: " + status);
            }
        } while (false);
    }


    public void FatalError(String s) {
        do {
            String _where, _message;
            _message = s + "\n";
            System.out.println(_message + "\nAborting...\n");
            jcuda.runtime.JCuda.cudaDeviceReset();
        } while (false);
    }

    public static void main(String[] args) {

        CublasUtil.startup();

        Random rand = new Random(1);
        float[][] Aarray = a.randFloat(2, 3, rand);
        float[][] Barray = a.randFloat(2, 3, rand);

        {
            System.out.println("\n\nCPU");
            FloatMatrix A = new FloatMatrix(Aarray);
            System.out.println(a.toString(A.transpose().toArray2()));
            FloatMatrix B = new FloatMatrix(Barray);
            FloatMatrix C = A.add(B.mul(-2.0f));
            System.out.println(a.toString(C.toArray2()));
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.build(Aarray);
            System.out.println(a.toString(A.transpose().toArray2()));
            Matrix B = Matrix.build(Barray);
            Matrix C = A.comb(1.0f, -2.0f, B);
            System.out.println(a.toString(C.toArray2()));
            A.free();
            B.free();
            C.free();
        }

        Aarray = new float[][] {{1, 2, 3}, {-1, -2.5f, 1.0f}};
        Barray = new float[][] {{4, 0, -1}, {0, -2.5f, 1}, {9, -10, -0.5f}};
        float[][] Carray = new float[][] {{1, 2, 3}, {1, 2, 3}};

        {
            System.out.println("\n\nCPU");
            FloatMatrix A = new FloatMatrix(Aarray);
            FloatMatrix B = new FloatMatrix(Barray);
            FloatMatrix C = new FloatMatrix(Carray);
            FloatMatrix D = A.mmul(B.transpose()).add(FloatMatrix.ones(2, 3)).muli(2.0f).mul(C);
            D.maxi(-68.0f);
            System.out.println(a.toString(D.toArray2()));
            System.out.println(D.norm1());
            System.out.println(D.norm2());
            FloatMatrix E = D.div(A);
            System.out.println(a.toString(E.toArray2()));
            System.out.println(E.norm1());
            System.out.println(E.norm2());
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.build(Aarray);
            Matrix B = Matrix.build(Barray);
            Matrix C = Matrix.build(Carray);
            Matrix D = (B.transpose().dot(A).add(Matrix.ones(2, 3)).muli(2.0f)).mul(C);
            D.maxi(-68.0f);
            System.out.println(a.toString(D.toArray2()));
            System.out.println(D.norm1());
            System.out.println(D.norm2());
            Matrix E = D.div(A);
            System.out.println(a.toString(E.toArray2()));
            System.out.println(E.norm1());
            System.out.println(E.norm2());
            A.free();
            B.free();
            C.free();
            D.free();
        }

        {
            System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
            System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
            System.out.println("\n\n"+a.toString(Matrix.ones(300, 3).toArray2()));
        }

        // Misc tests
        ////////////////////////////////////////////////

        {
            Matrix A = Matrix.ones(3, 3);
            A.muli(4.0f);
            System.out.println(a.toString(A.toArray2()));
            System.out.println(a.toString(A.sqrt().toArray2()));
            System.out.println(a.toString(A.toArray2()));
            A.sqrti();
            System.out.println(a.toString(A.toArray2()));
            A.free();
        }

        {
            int b=10;
            List<Matrix> A = new ArrayList<Matrix>();
            List<Matrix> B = new ArrayList<Matrix>();
            for (int i=0; i<b; ++i) {
                A.add(Matrix.build(a.randFloat(3, 3, rand)));
                B.add(Matrix.build(a.randFloat(3, 3, rand)));
            }
            System.out.println("CPU:");
            for (int i=0; i<b; ++i) {
                FloatMatrix Amat = new FloatMatrix(A.get(i).toArray2());
                FloatMatrix Bmat = new FloatMatrix(B.get(i).toArray2());
                System.out.println(a.toString(Amat.mmul(Bmat).toArray2()));
            }
            System.out.println("GPU:");
            List<Matrix> C = Matrix.mmul(A, B);
            for (int i=0; i<b; ++i) {
                System.out.println(a.toString(C.get(i).toArray2()));
            }

            System.out.println("CPU:");
            for (int i=0; i<b; ++i) {
                FloatMatrix Amat = new FloatMatrix(A.get(i).toArray2());
                FloatMatrix inv = Solve.solve(Amat, FloatMatrix.eye(Amat.rows));
                System.out.println(a.toString(inv.toArray2()));
            }
            System.out.println("GPU:");
            C = Matrix.invert(A);
            for (int i=0; i<b; ++i) {
                System.out.println(a.toString(C.get(i).toArray2()));
            }
        }

        {
            System.out.println("\n\nCPU");
            FloatMatrix A = new FloatMatrix(Aarray);
            System.out.println(a.toString(A.rowSums().toArray2()));
            System.out.println(a.toString(A.columnSums().toArray2()));
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.build(Aarray);
            System.out.println(a.toString(A.colSum().toArray2()));
            System.out.println(a.toString(A.rowSum().toArray2()));
            A.free();
        }

        {
            System.out.println("\n\nCPU");
            FloatMatrix A = new FloatMatrix(Aarray);
            FloatMatrix col = FloatMatrix.ones(2, 1);
            col.put(0, 0, 2.0f);
            FloatMatrix row = FloatMatrix.ones(1, 3);
            row.put(0, 0, 2.0f);
            System.out.println(a.toString(A.addColumnVector(col).toArray2()));
            System.out.println(a.toString(A.addRowVector(row).toArray2()));
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.build(Aarray);
            float[][] col = a.onesFloat(2, 1);
            col[0][0] = 2.0f;
            float[][] row = a.onesFloat(1, 3);
            row[0][0] = 2.0f;
            System.out.println(a.toString(A.colAdd(Matrix.build(col)).toArray2()));
            System.out.println(a.toString(A.rowAdd(Matrix.build(row)).toArray2()));
            A.free();
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.rand(5, 7, rand);
            System.out.println(a.toString(A.toArray2()));
            System.out.println(a.toString(A.copySubmatrix(1, 3, 2, 4).toArray2()));
            A.setSubmatrix(Matrix.ones(2,2), 1, 2);
            System.out.println(a.toString(A.toArray2()));
            A.setSubmatrix(a.onesFloat(2,2), 1, 0);
            System.out.println(a.toString(A.toArray2()));
            A.set(4, 3, 5.0f);
            System.out.println(a.toString(A.toArray2()));
            Matrix B = Matrix.rand(2, 3, rand);
            System.out.println(a.toString(B.toArray2()));
            A.setSubmatrix(1, 1, B, 1, 2, 1, 3);
            System.out.println(a.toString(A.toArray2()));
            A.free();
        }

        {
            System.out.println("\n\nCPU");
            FloatMatrix A = new FloatMatrix(Aarray);
            FloatMatrix col = FloatMatrix.ones(2, 1);
            col.put(0, 0, 2.0f);
            FloatMatrix row = FloatMatrix.ones(1, 3);
            row.put(0, 0, 2.0f);
            System.out.println(a.toString(A.toArray2()));
            System.out.println(a.toString(A.mulColumnVector(col).toArray2()));
            System.out.println(a.toString(A.mulRowVector(row).toArray2()));
            System.out.println(a.toString(A.divColumnVector(col).toArray2()));
            System.out.println(a.toString(A.divRowVector(row).toArray2()));
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.build(Aarray);
            float[][] col = a.onesFloat(2, 1);
            col[0][0] = 2.0f;
            float[][] row = a.onesFloat(1, 3);
            row[0][0] = 2.0f;
            System.out.println(a.toString(A.toArray2()));
            System.out.println(a.toString(A.colMul(Matrix.build(col)).toArray2()));
            System.out.println(a.toString(A.rowMul(Matrix.build(row)).toArray2()));
            System.out.println(a.toString(A.colDiv(Matrix.build(col)).toArray2()));
            System.out.println(a.toString(A.rowDiv(Matrix.build(row)).toArray2()));
            A.rowDivi(Matrix.build(row));
            System.out.println(a.toString(A.toArray2()));

            A.free();
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.ones(5, 5);
            System.out.println(a.toString(A.toArray2()));
            System.out.println(a.toString(A.diagAdd(10.0f).toArray2()));
            A.diagAddi(1.0f);
            System.out.println(a.toString(A.toArray2()));

            A.free();
        }

        {
            int b=3;
            List<Matrix> A = new ArrayList<Matrix>();
            List<Matrix> B = new ArrayList<Matrix>();
            for (int i=0; i<b; ++i) {
                A.add(Matrix.build(a.randFloat(3, 3, rand)));
                B.add(Matrix.build(a.randFloat(3, 3, rand)));
            }
            System.out.println("CPU:");
            for (int i=0; i<b; ++i) {
                FloatMatrix Amat = new FloatMatrix(A.get(i).toArray2());
                FloatMatrix inv = Solve.solve(Amat, FloatMatrix.eye(Amat.rows));
                System.out.println(a.toString(inv.toArray2()));
            }
            System.out.println("GPU:");
            List<Matrix> C = Matrix.invert(A);
            for (int i=0; i<b; ++i) {
                System.out.println(a.toString(C.get(i).toArray2()));
            }
        }

        {
            System.out.println("\n\nGPU");
            Matrix A = Matrix.ones(5, 5);
            A.muli(-0.2f);
            System.out.println(a.toString(A.toArray2()));
            A.powi(1.0f);
            System.out.println(a.toString(A.toArray2()));

            A.free();
        }

        CublasUtil.shutdown();
    }

}