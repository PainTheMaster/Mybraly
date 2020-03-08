package matrixbasic

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

