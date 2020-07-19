package deeplearning

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"fmt"
)

//NeuralNet is a neural network composed of a weight matrix "W", bias vector "B",
//activation function for hdden layers "ActivFuncHidden", activation function for out-put layer "ActivFuncOut".
type NeuralNet struct {
	W               [][][]float64
	B               [][]float64
	Midval          []linearalgebra.Colvec
	Output          []linearalgebra.Colvec
	ActivFuncHidden func(float64) float64
	ActivFuncOut    func(linearalgebra.Colvec) linearalgebra.Colvec
}

//Make makes a new empty nerral network "neuralNet". "nodes" represents the number of nodes in each layer
func Make(nodes []int, activFuncHidden func(float64) float64, activFuncOut func(linearalgebra.Colvec) linearalgebra.Colvec) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1)
		}
		neuralNet.Midval[i] = make(linearalgebra.Colvec, nodes[i])
		neuralNet.Output[i] = make(linearalgebra.Colvec, nodes[i]+1)
		neuralNet.Output[i][nodes[i]] = 1.0
	}

	neuralNet.ActivFuncHidden = activFuncHidden
	neuralNet.ActivFuncOut = activFuncOut

	return
}

//Forward calculates output of a neural "neuralNet" from the input "input".
func (neuralNet NeuralNet) Forward(input linearalgebra.Colvec) (output linearalgebra.Colvec) {
	if len(neuralNet.W[1][0]) != len(input)+1 {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	neuralNet.Output[0] = append(input, 1.0)

	for layer := 1; layer <= len(neuralNet.W)-2; layer++ {
		neuralNet.Midval[layer] = linearalgebra.MatColvecMult(neuralNet.W[layer], neuralNet.Output[layer-1])
		for i := range neuralNet.Midval[layer] {
			neuralNet.Output[layer][i] = neuralNet.ActivFuncHidden(neuralNet.Midval[layer][i])
		}
	}

	{
		layer := len(neuralNet.W) - 1
		neuralNet.Midval[layer] = linearalgebra.MatColvecMult(neuralNet.W[layer], neuralNet.Output[layer-1])
		neuralNet.Output[layer] = neuralNet.ActivFuncOut(neuralNet.Midval[layer])
		output = neuralNet.Output[layer]
	}

	return
}
