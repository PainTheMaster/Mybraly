package deeplearning

import (
	"PainTheMaster/mybraly/service"
	"math"
)

//Sigmoid returns 1/(1+Exp(-x))
func Sigmoid(x float64) (ans float64) {
	ans = 1.0 / (1.0 + math.Exp(-1.0*x))
	return
}

//Step returns 1 if x >= 0, 0 if x < 0
func Step(x float64) (ans float64) {
	if x >= 0.0 {
		ans = 1.0
	} else {
		ans = 0.0
	}
	return
}

//ReLU is Reflected linear unit: this returns x if x >= 0, 0 if x < 0.
func ReLU(x float64) (ans float64) {
	if x >= 0.0 {
		ans = x
	} else {
		ans = 0.0
	}
	return
}

//SoftMax is the soft max function: exp(x_n)/sum(exp(x_i))
func SoftMax(x []float64) (ans []float64) {
	length := len(x)
	ans = make([]float64, length)

	maxVal, _ := service.MaxFloat64(x)

	if math.Log(float64(length))+maxVal > math.Log(math.MaxFloat64) {
		p := math.Log(float64(length)) + maxVal - math.Log(math.MaxFloat64) + 0.1 //1.0 is just for redundancy
		var sum float64
		sum = 0.0
		for i := range x {
			sum += math.Exp(x[i] - p)
		}
		for i := range x {
			ans[i] = math.Exp(x[i]-p) / sum
		}
	} else {
		var sum float64
		sum = 0.0
		for i := range x {
			sum += math.Exp(x[i])
		}
		for i := range x {
			ans[i] = math.Exp(x[i]) / sum
		}
	}
	return
}

//SigmaOutput is an identity function for output.
func SigmaOutput(in []float64) (out []float64) {
	out = in
	return
}
