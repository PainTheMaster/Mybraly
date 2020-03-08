package eigen

import "math"

//EigenVecByQR calculates all eigen vectors of a square matix A by using reverse iteration method.
//"iter" is the iteration number of repetition.
//Each row in the answer "eigenVes" is a eigen vector.
func EigenVecByQR(A [][]float64, iter int) (eigenVecs [][]float64) {
	//サイズを求める。
	size := len(A)

	eigenVecs = make([][]float64, size)
	for i := range eigenVecs {
		eigenVecs[i] = make([]float64, size)
	}

	//固有値を求めて、ベクトルに保持しておく。
	RQ := EigenValByQR(A, iter)
	eigenVal := make([]float64, size)
	for i := 0; i <= size-1; i++ {
		eigenVal[i] = RQ[i][i]
	}

	B := make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		B[i] = make([]float64, size)
		for j := 0; j <= size-1; j++ {
			B[i][j] = A[i][j]
		}
	}

	//固有値シフトのデルタを求める。
	deltaNeg, deltaPos := helperLambdaDelta(eigenVal)

	for idxRambda := 0; idxRambda <= size-1; idxRambda++ {
		var shift float64
		if eigenVal[idxRambda] >= 0 {
			shift = eigenVal[idxRambda] + deltaPos
		} else {
			shift = eigenVal[idxRambda] - deltaNeg
		}

		for j := 0; j <= size-1; j++ {
			B[j][j] = A[j][j] - shift
		}

		//正は(固有値+デルタ)を引く。負は(固有値-デルタ)を引く。なお、デルタはいずれの場合も正
		//固有値シフトした行列の逆行列をLU法で求める。

		for j := range eigenVecs[idxRambda] {
			eigenVecs[idxRambda][j] = 1.0
		}

		for round := 0; round <= iter-1; round++ {
			var norm float64

			eigenVecs[idxRambda] = LuSolver(B, eigenVecs[idxRambda])

			norm = 0.0
			for k := 0; k <= size-1; k++ {
				norm += eigenVecs[idxRambda][k] * eigenVecs[idxRambda][k]
			}

			scale := 1.0 / math.Sqrt(norm)
			for k := range eigenVecs[idxRambda] {
				eigenVecs[idxRambda][k] = eigenVecs[idxRambda][k] * scale
			}
		}
	}
	return
}

//EigenValByQR calculates eigen value of a non-symmetric matrix A by using QR transformation.
//"iter" is iteration number. "Ans" is an upper triangular matrix and the diagonal elements are the eigen values.
func EigenValByQR(A [][]float64, iter int) (Ans [][]float64) {
	size := len(A)

	R := make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		R[i] = make([]float64, size)
		for j := 0; j <= size-1; j++ {
			R[i][j] = A[i][j]
		}
	}

	var Qt [][]float64

	for i := 1; i <= iter; i++ {
		/*まずQt,Rを求める*/
		Qt, R = Qr(R) /*宣言同時代入(:=)してはいけない。そうしてしまう右辺と左辺のRが別々のシンボルとなり、毎回Rの初期値から初めてしまうようだ!*/

		/* Qを順に掛けていく */
		lenQt := len(Qt)
		for idxQt := 0; idxQt <= lenQt-1; idxQt++ {
			for row := 0; row <= size-1; row++ {
				helperMultiplyQrHorizon(R, row, Qt[idxQt])
			}
		}

	}

	Ans = R
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
			helperMultiplyQrVertical(R, colProduct, tempH)
		}

	}

	return
}

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

//LU decomposes a square matrix A into L and U. Diagonal elements of L are 1
func lu(A [][]float64) (LU [][]float64) {
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
	LU := lu(A)
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

//LuSolver solves a equasion Ax=y by using LU decomposition
func LuSolver(A [][]float64, y []float64) (x []float64) {
	size := len(A)
	x = make([]float64, size)
	interm := make([]float64, size)

	LU := lu(A)
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

//TripDiagHouseholder transforms a square matrix A into a triple diagonal matix TripDiag
//this function also returns the "Householder vectors"
func TripDiagHouseholder(A [][]float64) (householderVec [][]float64, TripDiag [][]float64) {
	size := len(A)

	tempMat := make([][]float64, size)
	for i := 0; i <= size-1; i++ {
		tempMat[i] = make([]float64, size)
		for j := 0; j <= size-1; j++ {
			tempMat[i][j] = A[i][j]
		}
	}

	householderVec = make([][]float64, size-2)
	for i := 0; i <= size-3; i++ {
		householderVec[i] = make([]float64, size-1-i)
	}

	TripDiag = make([][]float64, 2)
	TripDiag[0] = make([]float64, size)
	TripDiag[1] = make([]float64, size-1)

	for piv := 0; piv <= size-3; piv++ {
		TripDiag[0][piv] = tempMat[piv][piv]
		householderVec[piv], TripDiag[1][piv] = Householder(tempMat, piv, piv+1)
		tempMat[piv+1][piv] = TripDiag[1][piv]
		for row := piv + 2; row <= size-1; row++ {
			tempMat[row][piv] = 0.0
		}
		for colMulti := piv + 1; colMulti <= size-1; colMulti++ {
			helperMultiplyQrVertical(tempMat, colMulti, householderVec[piv])
		}

		tempMat[piv][piv+1] = TripDiag[1][piv]
		for col := piv + 2; col <= size-1; col++ {
			tempMat[piv][col] = 0.0
		}
		for colMulti := piv + 1; colMulti <= size-1; colMulti++ {
			helperMultiplyQrHorizon(tempMat, colMulti, householderVec[piv])
		}
	}

	TripDiag[0][size-2] = tempMat[size-2][size-2]
	TripDiag[1][size-2] = tempMat[size-1][size-2]

	TripDiag[0][size-1] = tempMat[size-1][size-1]

	return
}

//EigenValSymmByGivens calculates eigen values of a symmetric matrix A.
//"iter" is the iteration number of calculation
func EigenValSymmByGivens(A [][]float64, iter int) (eigenVal []float64) {
	size := len(A)
	_, TripDiag := TripDiagHouseholder(A)

	cos := make([]float64, size-1)
	sin := make([]float64, size-1)

	normCosSin := func(piv int) (norm, cos, sin float64) {
		norm = TripDiag[0][piv]*TripDiag[0][piv] + TripDiag[1][piv]*TripDiag[1][piv]
		norm = math.Sqrt(norm)
		cos = TripDiag[0][piv] / norm
		sin = TripDiag[1][piv] / norm
		return
	}

	givensRotationLeft := func(piv int) {
		temp := cos[piv]*TripDiag[1][piv] + sin[piv]*TripDiag[0][piv+1]
		TripDiag[0][piv+1] = -sin[piv]*TripDiag[1][piv] + cos[piv]*TripDiag[0][piv+1]
		TripDiag[1][piv] = temp
	}

	givensRotationRight := func(piv int) {
		TripDiag[0][piv] = cos[piv]*TripDiag[0][piv] + sin[piv]*TripDiag[1][piv]
		TripDiag[1][piv] = sin[piv] * TripDiag[0][piv+1]
		TripDiag[0][piv+1] = cos[piv] * TripDiag[0][piv+1]
	}

	for cycle := 0; cycle <= iter-1; cycle++ {
		{
			piv := 0
			TripDiag[0][piv], cos[piv], sin[piv] = normCosSin(piv)
			givensRotationLeft(piv)
		}

		for piv := 1; piv <= size-2; piv++ {
			TripDiag[0][piv], cos[piv], sin[piv] = normCosSin(piv)
			TripDiag[1][piv] *= cos[piv-1]
			givensRotationLeft(piv)
		}

		for piv := 0; piv <= size-2; piv++ {
			givensRotationRight(piv)
		}
	}

	eigenVal = make([]float64, size)
	for piv := 0; piv <= size-1; piv++ {
		eigenVal[piv] = TripDiag[0][piv]
	}

	return
}
