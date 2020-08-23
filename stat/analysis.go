package stat

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"math"
)

//Normalize normalizes the data: average = 0, variance = 1.
func Normalize(data []float64) (normData []float64) {
	numData := len(data)
	ave := Average(data)
	stdev := StdevSamp(data)

	normData = make([]float64, numData)

	for i := range data {
		normData[i] = (data[i] - ave) / stdev
	}
	return
}

//NormalizeTable normalizes the data: average = 0, variance = 1 in all dimensions.
func NormalizeTable(data [][]float64) (normData [][]float64) {
	cols := len(data[0])
	rows := len(data)

	ave := AverageTable(data)
	stdev := StdevSampTable(data)

	normData = make([][]float64, rows)
	for i := range normData {
		normData[i] = make([]float64, cols)
		for j := range normData[i] {
			normData[i][j] = (data[i][j] - ave[j]) / stdev[j]
		}
	}
	return
}

//CovarianceMatrix calculates the covariance matrix of the data.
func CovarianceMatrix(data [][]float64) (covMatrix [][]float64) {
	dim := len(data[0])
	dataNum := len(data)
	covMatrix = make([][]float64, dim)
	for i := range covMatrix {
		covMatrix[i] = make([]float64, dim)
	}
	ave := AverageTable(data)
	//	variance := VarianceSampTable(data)

	for i := 0; i <= dim-1; i++ {
		for j := i; j <= dim-1; j++ {
			for n := 0; n <= dataNum-1; n++ {
				covMatrix[i][j] += (data[n][i] - ave[i]) * (data[n][j] - ave[j])
			}
			covMatrix[i][j] /= float64(dataNum - 1)
			covMatrix[j][i] = covMatrix[i][j]
		}
	}
	return
}

//CorrCoeffMatrix  calculates correlation coefficient matrix.
func CorrCoeffMatrix(data [][]float64) (corrMat [][]float64) {
	dim := len(data[0])
	corrMat = make([][]float64, dim)
	for i := range corrMat {
		corrMat[i] = make([]float64, dim)
	}
	covMat := CovarianceMatrix(data)
	for i := range covMat {
		corrMat[i][i] = 1.0
		for j := i + 1; j <= dim-1; j++ {
			corrMat[i][j] = covMat[i][j] / math.Sqrt(covMat[i][i]*covMat[j][j])
			corrMat[j][i] = corrMat[i][j]
		}
	}
	return
}

//PrincipleCompAnalysis performs principle component analysis.
func PrincipleCompAnalysis(data [][]float64) (eigenVal []float64, eigenVec [][]float64, factorLoad [][]float64) {
	dim := len(data[0])

	normTab := NormalizeTable(data)
	corMat := CorrCoeffMatrix(normTab)

	iteration := 100
	eigenValQR := linearalgebra.EigenValByQR(corMat, iteration)
	eigenVal = make([]float64, dim)
	for i := 0; i <= dim-1; i++ {
		eigenVal[i] = eigenValQR[i][i]
	}
	eigenVec = linearalgebra.EigenVecByQR(corMat, iteration)
	for i := range eigenVec {
		var c float64
		for j := range eigenVec[i] {
			c += math.Pow(eigenVec[i][j], 2.0)
		}
		c = 1.0 / math.Sqrt(c)
		for j := range eigenVec[i] {
			eigenVec[i][j] *= c
		}
	}

	factorLoad = make([][]float64, dim)
	for i := range factorLoad {
		factorLoad[i] = make([]float64, dim)
		rootEigen := math.Sqrt(eigenVal[i])
		for j := range factorLoad[i] {
			factorLoad[i][j] = rootEigen * eigenVec[i][j]
		}
	}

	return
}
