package matrix

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
