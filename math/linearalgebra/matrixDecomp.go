package linearalgebra

import (
	helper "PainTheMaster/mybraly/math/linearalgebra/matrixDecomp"
	"math"
)

//Householder gives "Householder vector" of A focusing on the column c "col" and topVal. topVal is the 1st element of the transformed vector and corresponds to the norm of the original vector
//processes the column "col", from row "row" to size-1.
func Householder(A [][]float64, col int, row int) (h []float64, topVal float64) {
	size := len(A)

	topVal = 0.0
	for i := row; i <= size-1; i++ {
		topVal += A[i][col] * A[i][col]
	}

	tempSqNormH := topVal

	topVal = math.Sqrt(topVal)
	if A[row][col] > 0 {
		topVal *= -1.0
	}

	h = make([]float64, size-row)

	h[0] = A[row][col] - topVal

	for i := 1; i <= size-row-1; i++ {
		h[i] = A[row+i][col]
	}

	tempSqNormH -= A[row][col] * A[row][col]
	tempSqNormH += h[0] * h[0]

	normalizFactor := 1.0 / math.Sqrt(tempSqNormH)

	for i := 0; i <= size-row-1; i++ {
		h[i] *= normalizFactor
	}

	return
}

//Qr performs QR decomposition of A and returns corresponding Householder "vectors" and a matrix R
func Qr(A [][]float64) (Qt [][]float64, R [][]float64) {

	size := len(A)

	Qt = make([][]float64, size-1)

	R = make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		R[i] = make([]float64, size)
	}
	for i := 0; i <= size-1; i++ {
		for j := 0; j <= size-1; j++ {
			R[i][j] = A[i][j]
		}
	}

	for colPiv := 0; colPiv <= size-2; colPiv++ {
		tempH, topVal := Householder(R, colPiv, colPiv)

		Qt[colPiv] = make([]float64, size-colPiv)
		for row := 0; row <= size-colPiv-1; row++ {
			Qt[colPiv][row] = tempH[row]
		}

		R[colPiv][colPiv] = topVal
		for i := colPiv + 1; i <= size-1; i++ {
			R[i][colPiv] = 0.0
		}

		for colProduct := colPiv + 1; colProduct <= size-1; colProduct++ {
			//			helper.HelperMultiplyQrVertical(R, colProduct, tempH)
			helper.HelperMultiplyQrVertical(R, colProduct, tempH)
		}

	}

	return
}

//Lu decomposes a square matrix A into L and U. Diagonal elements of L are 1
func Lu(A [][]float64) (LU [][]float64) {
	size := len(A)

	LU = make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		LU[i] = make([]float64, size)
	}

	for col := 0; col <= size-1; col++ {
		for row := 0; row <= col; row++ {
			var innerprod float64
			for i := 0; i <= row-1; i++ {
				innerprod += LU[row][i] * LU[i][col]
			}
			LU[row][col] = A[row][col] - innerprod
		}

		for row := col + 1; row <= size-1; row++ {
			var innerprod float64
			for i := 0; i <= row-1; i++ {
				innerprod += LU[row][i] * LU[i][col]
			}
			LU[row][col] = (A[row][col] - innerprod) / LU[col][col]
		}
	}
	return
}
