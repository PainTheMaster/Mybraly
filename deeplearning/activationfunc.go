package deeplearning

import (
	"PainTheMaster/mybraly/mymath/linearalgebra"
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

//LabelSigmoid is a label for Sigmoid
const LabelSigmoid = "Sigmoid"

//ReLU is a set of ReLU and its derivative
var ReLU ActFuncHiddenSet

//LabelReLU is a label for ReLU
const LabelReLU = "ReLU"

//Identity is a set of itendtity function and its derivative
var Identity ActFuncHiddenSet

//LabelIdentity is a label for Identity unit
const LabelIdentity = "Identity"

//SoftMax is a softmax activation function and its derivative.
var SoftMax ActFuncOutputSet

//LabelSoftMax is a label for Soft max
const LabelSoftMax = "SoftMax"

//InitActFunc makes the active function structures
func InitActFunc() (actfuncHidden map[string]ActFuncHiddenSet, actfuncOut map[string]ActFuncOutputSet) {
	actfuncHidden = make(map[string]ActFuncHiddenSet)
	actfuncOut = make(map[string]ActFuncOutputSet)
	Sigmoid = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x), len(x)+1)
			for i := range x {
				ans[i] = 1.0 / (1.0 + math.Exp(-1.0*x[i]))
			}
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
	actfuncHidden[LabelSigmoid] = Sigmoid

	ReLU = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x), len(x)+1)
			for i := range x {
				if x[i] >= 0.0 {
					ans[i] = x[i]
				} else {
					ans[i] = 0.0
				}
			}
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
	actfuncHidden[LabelReLU] = ReLU

	Identity = ActFuncHiddenSet{
		Forward: func(x linearalgebra.Colvec) (ans linearalgebra.Colvec) {
			ans = make(linearalgebra.Colvec, len(x), len(x)+1)
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
	actfuncHidden[LabelIdentity] = Identity

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
	actfuncOut[LabelSoftMax] = SoftMax

	return
}
