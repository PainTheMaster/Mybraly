package service

//MaxFloat64 finds maximum value in a slice x and returns its value and index.
func MaxFloat64(x []float64) (ansVal float64, ansIdx int) {
	ansIdx = 0
	ansVal = x[0]
	for idx, val := range x {
		if val > ansVal {
			ansVal = val
			ansIdx = idx
		}
	}
	return
}

//MinFloat64 finds minimum value in a slice x and returns its value and index.
func MinFloat64(x []float64) (ansVal float64, ansIdx int) {
	ansIdx = 0
	ansVal = x[0]
	for idx, val := range x {
		if val < ansVal {
			ansVal = val
			ansIdx = idx
		}
	}
	return
}
