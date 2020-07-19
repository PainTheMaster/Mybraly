package linearalgebra

import "fmt"

//Colvec is column vector
type Colvec []float64

//Rowvec is row vector
type Rowvec []float64

//MScale multiplies all elements of A a scale "scale"
func MScale(scale float64, A [][]float64) (Ans [][]float64) {
	rowsA := len(A)
	colsA := len(A[0])

	Ans = make([][]float64, rowsA)
	for row := 0; row <= rowsA-1; row++ {
		Ans[row] = make([]float64, colsA)
		for col := 0; col <= colsA-1; col++ {
			Ans[row][col] = scale * A[row][col]
		}
	}
	return
}

//MAdd adds matrices A and B
func MAdd(A, B [][]float64) (Ans [][]float64) {
	rowsA := len(A)
	colsA := len(A[0])

	Ans = make([][]float64, rowsA)
	for row := 0; row <= rowsA-1; row++ {
		Ans[row] = make([]float64, colsA)
		for col := 0; col <= colsA-1; col++ {
			Ans[row][col] = A[row][col] + B[row][col]
		}
	}
	return
}

//MSubtract returns A-B
func MSubtract(A, B [][]float64) (Ans [][]float64) {
	rowsA := len(A)
	colsA := len(A[0])

	Ans = make([][]float64, rowsA)
	for row := 0; row <= rowsA-1; row++ {
		Ans[row] = make([]float64, colsA)
	}

	Ans = MAdd(A, MScale(-1.0, B))

	return
}

//MMult calculates the matrix product AB
//col of A and row of B has to be equal
func MMult(A, B [][]float64) (Ans [][]float64) {
	rowsA := len(A)
	colsB := len(B[0])
	common := len(B)

	Ans = make([][]float64, rowsA)
	for i := 0; i <= rowsA-1; i++ {
		A[i] = make([]float64, colsB)
	}

	for row := 0; row <= rowsA-1; row++ {
		for col := 0; col <= colsB; col++ {
			var innerprod float64
			for i := 0; i <= common-1; i++ {
				innerprod += A[row][i] * B[i][col]
			}
			Ans[row][col] = innerprod
		}
	}

	return
}

//MatColvecMult calculates the multipulication of a matrix "A" and a column vector "v".
func MatColvecMult(A [][]float64, v Colvec) (ans Colvec) {
	rowsMat, colsMat := len(A), len(A[0])
	dimVec := len(v)
	if colsMat != dimVec {
		fmt.Println("An error in MatColvecMult():")
		fmt.Println("The number of columns of the matrix and the vector dimension don't match")
	}

	ans = make(Colvec, rowsMat)
	for i := range A {
		ans[i] = 0.0
		for j := range A[i] {
			ans[i] += A[i][j] * v[j]
		}
	}
	return
}

//RowvecMatMult calculates the multiplcation of a row vector "v" and a matrix "A"
func RowvecMatMult(v Rowvec, A [][]float64) (ans Rowvec) {
	rowsMat, colsMat := len(A), len(A[0])
	dimVec := len(v)
	if rowsMat != dimVec {
		fmt.Println("An error in RowvecMatMult():")
		fmt.Println("The number of columns of the matrix and the vector dimension don't match")
	}

	ans = make(Rowvec, colsMat)
	for j := range A[0] {
		ans[j] = 0.0
		for i := range A {
			ans[j] += v[i] * A[i][j]
		}
	}
	return
}
