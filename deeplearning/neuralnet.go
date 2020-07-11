package deeplearning

import (
	"PainTheMaster/mybraly/math/matrix"
	"fmt"
)

//NeuralNet is a neural network composed of a weight matrix "W", bias vector "B",
//activation function for hdden layers "ActivFuncHidden", activation function for out-put layer "ActivFuncOut".
type NeuralNet struct {
	W               [][][]float64
	B               [][]float64
	ActivFuncHidden func(float64) float64
	ActivFuncOut    func([]float64) []float64
}

//Make makes a new empty nerral network "neuralNet"
func Make(nodes []int, activFuncHidden func(float64) float64, activFuncOut func([]float64) []float64) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1])
		}
	}

	neuralNet.B = make([][]float64, layers)
	for i := 1; i <= layers-1; i++ {
		neuralNet.B[i] = make([]float64, nodes[i])
	}

	neuralNet.ActivFuncHidden = activFuncHidden
	neuralNet.ActivFuncOut = activFuncOut

	return
}

//Forward calculates output of a neural "neuralNet" from the input "input".
func (neuralNet NeuralNet) Forward(input []float64) (output []float64) {
	if len(neuralNet.W[1][0]) != len(input) {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	mid := input
	for layer := 1; layer <= len(neuralNet.W)-2; layer++ {
		mid = matrix.MatVecMult(neuralNet.W[layer], mid)
		for i := range mid {
			mid[i] += neuralNet.B[layer][i]
			mid[i] = neuralNet.ActivFuncHidden(mid[i])
		}
	}

	{
		layer := len(neuralNet.W) - 1
		mid = matrix.MatVecMult(neuralNet.W[layer], mid)
		for i := range mid {
			mid[i] += neuralNet.B[layer][i]
		}
	}
	output = neuralNet.ActivFuncOut(mid)
	return
}
