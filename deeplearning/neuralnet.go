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
	W     [][][]float64
	dW    [][][]float64
	diffW [][][]float64

	Midval        []linearalgebra.Colvec
	Output        []linearalgebra.Colvec
	Delta         []linearalgebra.Colvec
	ActFuncHidden []ActFuncHiddenSet
	ActivFuncOut  ActFuncOutputSet

	ParamMomentum struct {
		moment [][][]float64
	}

	ParamAdaGrad struct {
		rep   int
		SqSum [][][]float64
	}
}

//Make makes a new empty nerral network "neuralNet". "nodes" represents the number of nodes in each layer
func Make(nodes []int, strActFuncHidden []string, strActFuncOut string) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	neuralNet.dW = make([][][]float64, layers)
	neuralNet.diffW = make([][][]float64, layers)
	neuralNet.Midval = make([]linearalgebra.Colvec, layers)
	neuralNet.Output = make([]linearalgebra.Colvec, layers)
	neuralNet.ParamMomentum.moment = make([][][]float64, layers)
	neuralNet.ParamAdaGrad.SqSum = make([][][]float64, layers)

	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		neuralNet.dW[i] = make([][]float64, nodes[i])
		neuralNet.diffW[i] = make([][]float64, nodes[i])
		neuralNet.ParamMomentum.moment[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaGrad.SqSum[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1) //The last (nodes[i-1]-th) column is bias
			neuralNet.dW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.diffW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamMomentum.moment[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaGrad.SqSum[i][j] = make([]float64, nodes[i-1]+1)
		}

		neuralNet.Midval[i] = make(linearalgebra.Colvec, nodes[i])
		neuralNet.Output[i] = make(linearalgebra.Colvec, nodes[i]+1)
		neuralNet.Output[i][nodes[i]] = 1.0
	}

	neuralNet.Delta = make([]linearalgebra.Colvec, layers)
	for i := 1; i <= layers-1; i++ {
		neuralNet.Delta[i] = make(linearalgebra.Colvec, nodes[i])
	}

	actFuncHidden, actFuncOut := InitActFunc()
	for i := range strActFuncHidden {
		neuralNet.ActFuncHidden = append(neuralNet.ActFuncHidden, actFuncHidden[strActFuncHidden[i]])
	}
	neuralNet.ActivFuncOut = actFuncOut[strActFuncOut]

	//TODO: please implement initialization of W by using appropriate distribution.
	var seed int64 = 100
	randSource := rand.NewSource(seed)
	newRand := rand.New(randSource)
	randAve := 0.01
	for l := 1; l <= layers-2; l++ {
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l][0])
			randStdev := neuralNet.ActFuncHidden[l].StdevWtFunc(numBranch)
			for br := range neuralNet.W[l][node] {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randAve
			}
			if neuralNet.W[l][node][numBranch-1] < 0 {
				var idxMax int = numBranch - 2
				for i := numBranch - 2; i >= 0; i-- {
					if neuralNet.W[l][node][i] > neuralNet.W[l][node][idxMax] {
						idxMax = i
					}
				}
				neuralNet.W[l][node][numBranch-1], neuralNet.W[l][node][idxMax] = neuralNet.W[l][node][idxMax], neuralNet.W[l][node][numBranch-1]
			}
		}
	}
	{
		l := layers - 1
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l][0])
			randStdev := neuralNet.ActivFuncOut.StdevWtFunc(numBranch)
			for br := range neuralNet.W[l][node] {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randAve
			}
			if neuralNet.W[l][node][numBranch-1] < 0 {
				var idxMax int = numBranch - 2
				for i := numBranch - 2; i >= 0; i-- {
					if neuralNet.W[l][node][i] > neuralNet.W[l][node][idxMax] {
						idxMax = i
					}
				}
				neuralNet.W[l][node][numBranch-1], neuralNet.W[l][node][idxMax] = neuralNet.W[l][node][idxMax], neuralNet.W[l][node][numBranch-1]
			}
		}
	}

	neuralNet.ParamAdaGrad.rep = 0

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

//Differentiate calculates differentiation of error function dE/dw of the neuralnetwork based on a given dataset input and correct.
//The out put is averaged.
func (neuralNet *NeuralNet) Differentiate(input, correct []linearalgebra.Colvec) {
	numData := len(input)

	for layer := 1; layer <= len(neuralNet.diffW)-1; layer++ {
		for j := range neuralNet.diffW[layer] {
			for i := range neuralNet.diffW[layer][j] {
				neuralNet.diffW[layer][j][i] = 0.0
			}
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
					neuralNet.diffW[layer][j][i] += neuralNet.Delta[layer][j] * neuralNet.Output[layer-1][i]
				}
			}
		}
	}

	for layer := 1; layer <= len(neuralNet.diffW)-1; layer++ {
		for j := range neuralNet.diffW[layer] {
			for i := range neuralNet.diffW[layer][j] {
				neuralNet.diffW[layer][j][i] /= float64(numData)
			}
		}
	}
}

//GradDescent optimizes the neural network to fit the given dataset.
func (neuralNet *NeuralNet) GradDescent(input, correct []linearalgebra.Colvec, learnRate float64) (err float64) {
	numData := len(input)

	neuralNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuralNet.W)-1; layer++ {
		for j := range neuralNet.W[layer] {
			for i := range neuralNet.W[layer][j] {
				neuralNet.W[layer][j][i] -= neuralNet.diffW[layer][j][i] * learnRate
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

//AdaGrad performs AdaGrad optimization.
func (neuralNet *NeuralNet) AdaGrad(input, correct []linearalgebra.Colvec, learnRate float64) (err float64) {
	numData := len(input)

	if neuralNet.ParamAdaGrad.rep == 0 {
		smallNum := 1.0e-8
		neuralNet.GradDescent(input, correct, learnRate)
		for layer := 1; layer <= len(neuralNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuralNet.ParamAdaGrad.SqSum[layer])-1; j++ {
				for i := 0; i <= len(neuralNet.ParamAdaGrad.SqSum[layer][j])-1; i++ {
					neuralNet.ParamAdaGrad.SqSum[layer][j][i] = neuralNet.diffW[layer][j][i]*neuralNet.diffW[layer][j][i] + smallNum
				}
			}
		}
		neuralNet.ParamAdaGrad.rep++
	} else {
		neuralNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuralNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuralNet.ParamAdaGrad.SqSum[layer])-1; j++ {
				for i := 0; i <= len(neuralNet.ParamAdaGrad.SqSum[layer][j])-1; i++ {
					neuralNet.ParamAdaGrad.SqSum[layer][j][i] += neuralNet.diffW[layer][j][i] * neuralNet.diffW[layer][j][i]
					neuralNet.dW[layer][j][i] = -1.0 * learnRate * neuralNet.diffW[layer][j][i] / math.Sqrt(neuralNet.ParamAdaGrad.SqSum[layer][j][i])
					neuralNet.W[layer][j][i] += neuralNet.dW[layer][j][i]
				}
			}
		}
		neuralNet.ParamAdaGrad.rep++
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

	neuralNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuralNet.W)-1; layer++ {
		for j := range neuralNet.W[layer] {
			for i := range neuralNet.W[layer][j] {
				neuralNet.dW[layer][j][i] = momentRate*neuralNet.ParamMomentum.moment[layer][j][i] - (1.0-momentRate)*learnRate*neuralNet.diffW[layer][j][i]
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
