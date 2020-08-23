package stat

//Average calculates average of the data.
func Average(data []float64) (ave float64) {
	numData := len(data)
	for i := range data {
		ave += data[i]
	}
	ave /= float64(numData)
	return
}

//AverageTable calculates the average of the data in the table for all dimensions.
func AverageTable(data [][]float64) (ave []float64) {
	rows := len(data)
	cols := len(data[0])

	ave = make([]float64, cols)
	for j := 0; j <= cols-1; j++ {
		for i := 0; i <= rows-1; i++ {
			ave[j] += data[i][j]
		}
		ave[j] /= float64(rows)
	}
	return
}
