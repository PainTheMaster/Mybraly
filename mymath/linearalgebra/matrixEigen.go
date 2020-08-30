package linearalgebra

import (
	helper "PainTheMaster/mybraly/mymath/linearalgebra/matrixEigen"
	"math"
)

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
	deltaNeg, deltaPos := helper.HelperLambdaDelta(eigenVal)

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
				helper.HelperMultiplyQrHorizon(R, row, Qt[idxQt])
			}
		}

	}

	Ans = R
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
			helper.HelperMultiplyQrVertical(tempMat, colMulti, householderVec[piv])
		}

		tempMat[piv][piv+1] = TripDiag[1][piv]
		for col := piv + 2; col <= size-1; col++ {
			tempMat[piv][col] = 0.0
		}
		for colMulti := piv + 1; colMulti <= size-1; colMulti++ {
			helper.HelperMultiplyQrHorizon(tempMat, colMulti, householderVec[piv])
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
