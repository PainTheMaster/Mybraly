package stat

import (
	"fmt"
	"math"
)

//VarianceSamp calculates sample variance of the data in the table in all dimensions.
func VarianceSamp(data []float64) (variance float64) {
	numData := len(data)
	if numData <= 1 {
		fmt.Println("stat.VarianceSamp error: no variance defined for this data")
		variance = 0.0
		return
	}

	ave := Average(data)
	for i := range data {
		variance += math.Pow((data[i] - ave), 2.0)
	}
	variance /= float64(numData - 1)
	return
}

//StdevSamp calculates sample standard deviation of the data in all dimensions.
func StdevSamp(data []float64) (stdev float64) {
	variance := VarianceSamp(data)
	stdev = math.Sqrt(variance)
	return
}

//VarianceSampTable calculates variance of the data in the table in all dimensions.
func VarianceSampTable(data [][]float64) (variance []float64) {
	rows := len(data)
	cols := len(data[0])

	average := AverageTable(data)

	variance = make([]float64, cols)

	for j := 0; j <= cols-1; j++ {
		for i := 0; i <= rows-1; i++ {
			variance[j] += math.Pow(data[i][j]-average[j], 2.0)
		}
		variance[j] /= float64(rows - 1)
	}
	return
}

//StdevSampTable calculates sample standard deviation of the data in the table in all dimensions.
func StdevSampTable(data [][]float64) (stdev []float64) {
	stdev = make([]float64, len(data[0]))
	variance := VarianceSampTable(data)
	for j := range variance {
		stdev[j] = math.Sqrt(variance[j])
	}
	return
}
