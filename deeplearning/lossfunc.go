package deeplearning

import "math"

const (
	delta = 1.0e-7
)

//MeanSqError calculates mean square error between teacher data "t" and neural network's answer "y".
func MeanSqError(t, y []float64) (err float64) {
	err = 0.0
	for k := range t {
		err += math.Pow(y[k]-t[k], 2.0)
	}
	err /= 2.0
	return
}

//CrossEntropError calculates cross entropy error between teacher data "t" and neural network's answer "y".
func CrossEntropError(t, y []float64) (err float64) {
	err = 0.0
	for k := range t {
		err -= t[k] * math.Log(y[k]+delta)
	}
	return
}
