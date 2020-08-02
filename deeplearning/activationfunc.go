package deeplearning

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"PainTheMaster/mybraly/service"
	"math"
)

//ActFuncHiddenSet is a set of activation function. Forward is a activation function and Backward is the derivative of the activation function.
type ActFuncHiddenSet struct {
	Forward     func(x linearalgebra.Colvec) linearalgebra.Colvec
	Backward    func(x linearalgebra.Colvec, y linearalgebra.Colvec) linearalgebra.Colvec
	StdevWtFunc func(prevNode int) float64
}

//ActFuncOutputSet isa s set of activation function. Forward is a activation function and Backward is the derivative of the activation function.
type ActFuncOutputSet struct {
	Forward     func(x linearalgebra.Colvec) linearalgebra.Colvec
	Backward    func(x linearalgebra.Colvec, y linearalgebra.Colvec) linearalgebra.Colvec
	StdevWtFunc func(prevNode int) float64
}

//Sigmoid is a set of sigmoid function and its derivative
var Sigmoid ActFuncHiddenSet

//ReLU is a set of ReLU and its derivative
var ReLU ActFuncHiddenSet

//Identity is a set of itendtity function and its derivative
var Identity ActFuncHiddenSet

var SoftMax ActFuncOutputSet

//InitActFunc makes the active function structures
func InitActFunc() {
	Sigmoid = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x)+1)
			for i := range x {
				ans[i] = 1.0 / (1.0 + math.Exp(-1.0*x[i]))
			}
			ans[len(x)] = 1.0
			return
		},
		Backward: func(x linearalgebra.Colvec, y linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x))
			for i := range x {
				ans[i] = math.Exp(-1.0*x[i]) * y[i] * y[i]
			}
			return
		},
		StdevWtFunc: func(prevNode int) (ans float64) {
			ans = math.Sqrt(1.0 / float64(prevNode))
			return
		},
	}

	ReLU = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x)+1)
			for i := range x {
				if x[i] >= 0.0 {
					ans[i] = x[i]
				} else {
					ans[i] = 0.0
				}
			}
			ans[len(x)] = 1.0
			return
		},
		Backward: func(x linearalgebra.Colvec, y linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x)+1)
			for i := range x {
				if x[i] >= 0.0 {
					ans[i] = 1.0
				} else {
					ans[i] = 0.0
				}
			}
			return
		},
		StdevWtFunc: func(prevNode int) (ans float64) {
			ans = math.Sqrt(2.0 / float64(prevNode))
			return
		},
	}

	Identity = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x)+1)
			for i := range x {
				ans[i] = x[i]
			}
			return
		},
		Backward: func(x linearalgebra.Colvec, y linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x))
			for i := range x {
				ans[i] = x[i]
			}
			return
		},
		StdevWtFunc: func(prevNode int) (ans float64) {
			ans = math.Sqrt(1.0 / float64(prevNode))
			return
		},
	}

	SoftMax = ActFuncOutputSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			length := len(x)
			ans = make(linearalgebra.Colvec, length)

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
		},
		StdevWtFunc: func(prevNode int) (ans float64) {
			ans = math.Sqrt(1.0 / float64(prevNode))
			return
		},
	}
}

//SoftMax is the soft max function: exp(x_n)/sum(exp(x_i))
/*
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
*/

/*SigmaOutput is an identity function for output.
func SigmaOutput(in []float64) (out []float64) {
	out = in
	return
}*/
