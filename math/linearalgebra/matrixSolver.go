package linearalgebra

/*QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR*/
/*              QR related functions from here               */
/*QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR QR*/

//QrSolver gives solution to a equation QtRx=y
//A matrix has to be decomposed into a "Householder""Qt"
func QrSolver(Qt [][]float64, R [][]float64, y []float64) (x []float64) {

	size := len(R)

	yTemp := make([]float64, size)
	for i := 0; i <= size-1; i++ {
		yTemp[i] = y[i]
	}

	for idxQ := 0; idxQ <= size-2; idxQ++ {
		innerProd := 0.0
		for i := 0; i+idxQ <= size-1; i++ {
			innerProd += Qt[idxQ][i] * yTemp[idxQ+i]
		}

		for i := 0; i+idxQ <= size-1; i++ {
			yTemp[idxQ+i] = yTemp[idxQ+i] - 2.0*Qt[idxQ][i]*innerProd
		}
	}

	x = make([]float64, size)

	for rowSolv := size - 1; rowSolv >= 0; rowSolv-- {
		var sum float64
		for i := size - 1; i >= rowSolv+1; i-- {
			sum += R[rowSolv][i] * x[i]
		}
		x[rowSolv] = (yTemp[rowSolv] - sum) / R[rowSolv][rowSolv]
	}
	return
}

/*              QR related functions to here               */

/*LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU*/
/*              LU related functions to here               */
/*LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU LU*/

//LuSolver solves a equasion Ax=y by using LU decomposition
func LuSolver(A [][]float64, y []float64) (x []float64) {
	size := len(A)
	x = make([]float64, size)
	interm := make([]float64, size)

	LU := Lu(A)
	//L*interm = y, interm = Ux
	for i := 0; i <= size-1; i++ {
		var innerprod float64
		for j := 0; j <= i-1; j++ {
			innerprod += LU[i][j] * interm[j]
		}
		interm[i] = y[i] - innerprod
	}
	//Ux = interm
	for i := size - 1; i >= 0; i-- {
		var innerprod float64
		for j := size - 1; j >= i+1; j-- {
			innerprod += LU[i][j] * x[j]
		}
		x[i] = (interm[i] - innerprod) / LU[i][i]
	}
	return
}

//LuInverse calculates inverse matrices of LU matrices
//Diagonal elements of L are 1
func LuInverse(LU [][]float64) (LUinv [][]float64) {

	size := len(LU)

	LUinv = make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		LUinv[i] = make([]float64, size)
	}

	//U
	for col := size - 1; col >= 0; col-- {
		row := col
		LUinv[col][col] = 1.0 / LU[col][col]

		for row = col - 1; row >= 0; row-- {
			var innerprod float64
			for i := row + 1; i <= col; i++ {
				innerprod += LU[row][i] * LUinv[i][col]
			}
			LUinv[row][col] = -1.0 * innerprod / LU[row][row]
		}
	}

	//L
	for row := size - 1; row >= 0; row-- {

		for col := row - 1; col >= 0; col-- {
			var innerprod float64
			for i := col + 1; i <= row-1; i++ {
				innerprod += LUinv[row][i] * LU[i][col]
			}
			innerprod += 1.0 * LU[row][col]
			LUinv[row][col] = -1.0 * innerprod
		}
	}
	return
}

//InverseMatrix calculates a inverse matrix of A by using LU decompostition
func InverseMatrix(A [][]float64) (AInv [][]float64) {
	size := len(A)
	AInv = make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		AInv[i] = make([]float64, size)
	}
	LU := Lu(A)
	LUinv := LuInverse(LU)

	//UL
	for row := 0; row <= size-1; row++ {
		//colが小さい時（長い列ベクトルでUが制限因子になっている時）
		for col := 0; col <= row-1; col++ {
			var innerprod float64
			for i := row; i <= size-1; i++ {
				innerprod += LUinv[row][i] * LUinv[i][col]
			}
			AInv[row][col] = innerprod
		}

		for col := row; col <= size-1; col++ {
			var innerprod float64
			innerprod = LUinv[row][col] * 1.0
			for i := col + 1; i <= size-1; i++ {
				innerprod += LUinv[row][i] * LUinv[i][col]
			}
			AInv[row][col] = innerprod
		}
	}

	return
}
