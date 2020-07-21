package deeplearning

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"PainTheMaster/mybraly/service"
	"math"
)

//ActFuncHiddenSet is a set of activation function. Forward is a activation function and Backward is the derivative of the activation function.
type ActFuncHiddenSet struct {
	Forward  func(x float64) float64
	Backward func(x float64, y float64) float64
}

//ActFuncOutputSet isa s set of activation function. Forward is a activation function and Backward is the derivative of the activation function.
type ActFuncOutputSet struct {
	Forward  func(x linearalgebra.Colvec) linearalgebra.Colvec
	Backward func(x linearalgebra.Colvec, y linearalgebra.Colvec) linearalgebra.Colvec
}

//Sigmoid is a set of sigmoid function and its derivative
var Sigmoid ActFuncHiddenSet

//ReLU is a set of ReLU and its derivative
var ReLU ActFuncHiddenSet

//Identity is a set of itendtity function and its derivative
var Identity ActFuncHiddenSet

//InitActFunc makes the active function structures
func InitActFunc() {
	Sigmoid = ActFuncHiddenSet{
		Forward: func(x float64) (ans float64) {
			ans = 1.0 / (1.0 + math.Exp(-1.0*x))
			return
		},
		Backward: func(x float64, y float64) (ans float64) {
			ans = math.Exp(-1.0*x) * y * y
			return
		},
	}

	ReLU = ActFuncHiddenSet{
		Forward: func(x float64) (ans float64) {
			if x >= 0.0 {
				ans = x
			} else {
				ans = 0.0
			}
			return
		},
		Backward: func(x float64, y float64) (ans float64) {
			if x >= 0.0 {
				ans = 1.0
			} else {
				ans = 0.0
			}
			return
		},
	}

	Identity = ActFuncHiddenSet{
		Forward: func(x float64) (ans float64) {
			ans = x
			return
		},
		Backward: func(x float64, y float64) (ans float64) {
			ans = 1
			return
		},
	}
}

/*
func Step(x float64) (ans float64) {
	if x >= 0.0 {
		ans = 1.0
	} else {
		ans = 0.0
	}
	return
}
*/

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

/*SigmaOutput is an identity function for output.
func SigmaOutput(in []float64) (out []float64) {
	out = in
	return
}*/
