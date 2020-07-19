package helpermatrixeigen

import "math"

func HelperLambdaDelta(v []float64) (deltaNeg, deltaPos float64) {

	size := len(v)
	var count, previous int
	var gap float64

	//Positive
	gap = math.Abs(v[0] - v[size-1]) //ありうる最大のギャップ
	//小さい方から大きい方にスキャンしていく
	previous = 0
	for i := size - 1; i >= 0; i-- {
		if v[i] >= 0.0 && i < previous {
			count++
			//ギャップを計算して必要に応じて更新する。
			temp := v[i] - v[previous]
			if temp < gap {
				gap = temp
			}
			//previousを更新する。
			previous = i
		} else if v[i] >= 0.0 {
			count++
			previous = i
		}
	}

	if count == 0 {
		deltaPos = -1.0
	} else if count == 1 {
		deltaPos = v[previous] * 0.1
	} else {
		deltaPos = gap / 3.0
	}

	//Negative

	gap = math.Abs(v[0] - v[size-1]) //ありうる最大のギャップ
	count = 0
	previous = 0

	for i := size - 1; i >= 0; i-- {
		if v[i] < 0.0 && i < previous {
			count++
			//ギャップを計算して必要に応じて更新する。
			temp := v[previous] - v[i]
			if temp < gap {
				gap = temp
			}

			//previousを更新する。
			previous = i
		} else if v[i] < 0.0 {
			count++
			previous = i
		}
	}

	if count == 0 {
		deltaNeg = -1.0
	} else if count == 1 {
		deltaNeg = v[previous] * (-0.1)
	} else {
		deltaNeg = gap / 3.0
	}

	return
}

func HelperMultiplyQrVertical(R [][]float64, col int, v []float64) {
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
func HelperMultiplyQrHorizon(R [][]float64, row int, vt []float64) {

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
