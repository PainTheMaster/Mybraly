package matrix

func helperMultiplyQrVertical(R [][]float64, col int, v []float64) {
	sizeMat := len(R)
	sizeVec := len(v)

	var innerprod float64
	rowTop := sizeMat - sizeVec

	innerprod = 0.0
	for i := 0; i <= sizeVec-1; i++ {
		innerprod += R[rowTop+i][col] * v[i]
	}

	for i := 0; i <= sizeVec-1; i++ {
		R[rowTop+i][col] = R[rowTop+i][col] - 2.0*v[i]*innerprod
	}
}

//helperMultiplyQrhorizon multiplies a vector to a matrix from right in a Householder manner
func helperMultiplyQrHorizon(R [][]float64, row int, vt []float64) {

	sizeMat := len(R)
	sizeVec := len(vt)

	var innerprod float64

	innerprod = 0.0
	for j := 0; j <= sizeVec-1; j++ {
		innerprod += R[row][sizeMat-sizeVec+j] * vt[j]
	}

	for j := 0; j <= sizeVec-1; j++ {
		R[row][sizeMat-sizeVec+j] = R[row][sizeMat-sizeVec+j] - 2.0*vt[j]*innerprod
	}
}
