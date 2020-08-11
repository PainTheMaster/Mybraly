package deeplearning

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"fmt"
	"math"
	"math/rand"
)

//NeuralNet is a neural network composed of a weight matrix "W", bias vector "B",
//activation function for hdden layers "ActivFuncHidden", activation function for out-put layer "ActivFuncOut".
type NeuralNet struct {
	W             [][][]float64
	Midval        []linearalgebra.Colvec
	Output        []linearalgebra.Colvec
	Delta         []linearalgebra.Colvec
	ActFuncHidden []ActFuncHiddenSet
	ActivFuncOut  ActFuncOutputSet

	dW [][][]float64

	ParamMomentum struct {
		moment [][][]float64
	}
}

//Make makes a new empty nerral network "neuralNet". "nodes" represents the number of nodes in each layer
func Make(nodes []int, activFuncHidden []ActFuncHiddenSet, activFuncOut ActFuncOutputSet) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	neuralNet.dW = make([][][]float64, layers)
	neuralNet.ParamMomentum.moment = make([][][]float64, layers)

	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		neuralNet.dW[i] = make([][]float64, nodes[i])
		neuralNet.ParamMomentum.moment[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1) //The last (nodes[i-1]-th) column is bias
			neuralNet.dW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamMomentum.moment[i][j] = make([]float64, nodes[i-1]+1)
		}

		neuralNet.Midval[i] = make(linearalgebra.Colvec, nodes[i])
		neuralNet.Output[i] = make(linearalgebra.Colvec, nodes[i]+1)
		neuralNet.Output[i][nodes[i]] = 1.0
	}

	neuralNet.Delta = make([]linearalgebra.Colvec, layers)
	for i := 1; i <= layers-1; i++ {
		neuralNet.Delta[i] = make(linearalgebra.Colvec, nodes[i])
	}

	neuralNet.ActFuncHidden = activFuncHidden
	neuralNet.ActivFuncOut = activFuncOut

	//TODO: please implement initialization of W by using appropriate distribution.
	var seed int64 = 100
	randSource := rand.NewSource(seed)
	newRand := rand.New(randSource)
	randBias := 0.01
	for l := 1; l <= layers-2; l++ {
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l-1])
			randStdev := activFuncHidden[l].StdevWtFunc(numBranch)
			for br := range neuralNet.W[l][node] {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randBias
			}
		}
	}
	{
		l := layers - 1
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l-1])
			randStdev := activFuncOut.StdevWtFunc(numBranch)
			for br := range neuralNet.W[l][node] {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randBias
			}
		}
	}
	return
}

//Forward calculates output of a neural "neuralNet" from the input "input".
func (neuralNet *NeuralNet) Forward(input linearalgebra.Colvec) {
	if len(neuralNet.W[1][0]) != len(input)+1 {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	neuralNet.Output[0] = append(input, 1.0)

	for layer := 1; layer <= len(neuralNet.W)-2; layer++ {
		neuralNet.Midval[layer] = linearalgebra.MatColvecMult(neuralNet.W[layer], neuralNet.Output[layer-1])

		neuralNet.Output[layer] = neuralNet.ActFuncHidden[layer].Forward(neuralNet.Midval[layer])

		neuralNet.Output[layer] = append(neuralNet.Output[layer], 1.0)
	}

	{
		layer := len(neuralNet.W) - 1
		neuralNet.Midval[layer] = linearalgebra.MatColvecMult(neuralNet.W[layer], neuralNet.Output[layer-1])
		neuralNet.Output[layer] = neuralNet.ActivFuncOut.Forward(neuralNet.Midval[layer])
	}

	return
}

func (neuralNet NeuralNet) Error(input linearalgebra.Colvec, correct linearalgebra.Colvec) (err float64) {
	layer := len(neuralNet.Output)
	output := neuralNet.Output[layer-1]
	err = 0.0
	for i := range output {
		err -= correct[i] * math.Log(output[i])
	}
	return
}

//GradDecent optimizes the neural network to fit the given dataset.
func (neuralNet *NeuralNet) GradDecent(input, correct []linearalgebra.Colvec, learnRate float64) (err float64) {
	numData := len(input)

	diffW := make([][][]float64, len(neuralNet.W))
	for layer := range diffW {
		diffW[layer] = make([][]float64, len(neuralNet.W[layer]))
		for j := range diffW[layer] {
			diffW[layer][j] = make([]float64, len(neuralNet.W[layer][j]))
		}
	}

	for data := 0; data <= numData-1; data++ {
		neuralNet.Forward(input[data])

		layer := len(neuralNet.W) - 1
		for j := range neuralNet.Output[layer] {
			neuralNet.Delta[layer][j] = neuralNet.Output[layer][j] - correct[data][j]
		}
		for layer--; layer >= 1; layer-- {
			actFuncDiff := neuralNet.ActFuncHidden[layer].Backward(neuralNet.Midval[layer], neuralNet.Output[layer])
			for j := range neuralNet.Delta[layer] {
				neuralNet.Delta[layer][j] = 0.0
				for k := range neuralNet.Delta[layer+1] {
					neuralNet.Delta[layer][j] += neuralNet.Delta[layer+1][k] * neuralNet.W[layer+1][k][j] * actFuncDiff[j]
				}
			}
		}

		for layer = 1; layer <= len(neuralNet.W)-1; layer++ {
			for j := range neuralNet.W[layer] {
				for i := range neuralNet.W[layer][j] {
					diffW[layer][j][i] += neuralNet.Delta[layer][j] * neuralNet.Output[layer-1][i]
				}
			}
		}
	}

	for layer := 1; layer <= len(neuralNet.W)-1; layer++ {
		for j := range neuralNet.W[layer] {
			for i := range neuralNet.W[layer][j] {
				neuralNet.W[layer][j][i] -= diffW[layer][j][i] / float64(numData) * learnRate
			}
		}
	}

	err = 0.0
	for data := range input {
		err += neuralNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//Momentum performs single run of momentum based optimization by using a (mini-) batch.
func (neuralNet *NeuralNet) Momentum(input, correct []linearalgebra.Colvec, learnRate, momentRate float64) (err float64) {
	numData := len(input)

	diffW := make([][][]float64, len(neuralNet.W))
	for layer := range diffW {
		diffW[layer] = make([][]float64, len(neuralNet.W[layer]))
		for j := range diffW[layer] {
			diffW[layer][j] = make([]float64, len(neuralNet.W[layer][j]))
		}
	}

	for data := 0; data <= numData-1; data++ {
		neuralNet.Forward(input[data])

		layer := len(neuralNet.W) - 1
		for j := range neuralNet.Output[layer] {
			neuralNet.Delta[layer][j] = neuralNet.Output[layer][j] - correct[data][j]
		}
		for layer--; layer >= 1; layer-- {
			actFuncDiff := neuralNet.ActFuncHidden[layer].Backward(neuralNet.Midval[layer], neuralNet.Output[layer])
			for j := range neuralNet.Delta[layer] {
				neuralNet.Delta[layer][j] = 0.0
				for k := range neuralNet.Delta[layer+1] {
					neuralNet.Delta[layer][j] += neuralNet.Delta[layer+1][k] * neuralNet.W[layer+1][k][j] * actFuncDiff[j]
				}
			}
		}

		for layer = 1; layer <= len(neuralNet.W)-1; layer++ {
			for j := range neuralNet.W[layer] {
				for i := range neuralNet.W[layer][j] {
					diffW[layer][j][i] += neuralNet.Delta[layer][j] * neuralNet.Output[layer-1][i]
				}
			}
		}
	}

	for layer := 1; layer <= len(neuralNet.W)-1; layer++ {
		for j := range neuralNet.W[layer] {
			for i := range neuralNet.W[layer][j] {
				neuralNet.dW[layer][j][i] = momentRate*neuralNet.ParamMomentum.moment[layer][j][i] - (1.0-momentRate)*learnRate*diffW[layer][j][i]/float64(numData)
				neuralNet.W[layer][j][i] += neuralNet.dW[layer][j][i]
				neuralNet.ParamMomentum.moment[layer][j][i] = neuralNet.dW[layer][j][i]
			}
		}
	}

	err = 0.0
	for data := range input {
		err += neuralNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}
