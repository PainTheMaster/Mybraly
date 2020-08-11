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

	ActFuncHidden []ActFuncHiddenSet
	ActivFuncOut  ActFuncOutputSet

	Midval []linearalgebra.Colvec
	Output []linearalgebra.Colvec
	Delta  []linearalgebra.Colvec

	ParamGradDecent struct {
		//Hyper parameters
		LearnRate float64
	}

	ParamMomentum struct {
		//Hyer parameters
		LearnRate  float64
		MomentRate float64
		//Working parameters
		moment [][][]float64
	}

	ParamAdaGrad struct {
		//Hyper parameters
		LearnRate float64
		//Working parameters
		rep   int
		sqSum [][][]float64
	}

	ParamRMSProp struct {
		//Hyper parameters
		LearnRate float64
		DecayRate float64
		//Working parameters
		rep     int
		expMvAv [][][]float64
	}
}

//Make makes a new empty nerral network "neuralNet". "nodes" represents the number of nodes in each layer
func Make(nodes []int, strActFuncHidden []string, strActFuncOut string) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	neuralNet.dW = make([][][]float64, layers)
	neuralNet.diffW = make([][][]float64, layers)
	neuralNet.ParamMomentum.moment = make([][][]float64, layers)
	neuralNet.ParamAdaGrad.sqSum = make([][][]float64, layers)
	neuralNet.ParamRMSProp.expMvAv = make([][][]float64, layers)

	neuralNet.Midval = make([]linearalgebra.Colvec, layers)
	neuralNet.Output = make([]linearalgebra.Colvec, layers)

	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		neuralNet.dW[i] = make([][]float64, nodes[i])
		neuralNet.diffW[i] = make([][]float64, nodes[i])
		neuralNet.ParamMomentum.moment[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaGrad.sqSum[i] = make([][]float64, nodes[i])
		neuralNet.ParamRMSProp.expMvAv[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1) //The last (nodes[i-1]-th) column is bias
			neuralNet.dW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.diffW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamMomentum.moment[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaGrad.sqSum[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamRMSProp.expMvAv[i][j] = make([]float64, nodes[i-1]+1)
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
func (neuNet *NeuralNet) Forward(input linearalgebra.Colvec) {
	if len(neuNet.W[1][0]) != len(input)+1 {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	neuNet.Output[0] = append(input, 1.0)

	for layer := 1; layer <= len(neuNet.W)-2; layer++ {
		neuNet.Midval[layer] = linearalgebra.MatColvecMult(neuNet.W[layer], neuNet.Output[layer-1])

		neuNet.Output[layer] = neuNet.ActFuncHidden[layer].Forward(neuNet.Midval[layer])

		neuNet.Output[layer] = append(neuNet.Output[layer], 1.0)
	}

	{
		layer := len(neuNet.W) - 1
		neuNet.Midval[layer] = linearalgebra.MatColvecMult(neuNet.W[layer], neuNet.Output[layer-1])
		neuNet.Output[layer] = neuNet.ActivFuncOut.Forward(neuNet.Midval[layer])
	}

	return
}

func (neuNet NeuralNet) Error(input linearalgebra.Colvec, correct linearalgebra.Colvec) (err float64) {
	layer := len(neuNet.Output)
	output := neuNet.Output[layer-1]
	err = 0.0
	for i := range output {
		err -= correct[i] * math.Log(output[i])
	}
	return
}

//Differentiate calculates differentiation of error function dE/dw of the neuralnetwork based on a given dataset input and correct.
//The out put is averaged.
func (neuNet *NeuralNet) Differentiate(input, correct []linearalgebra.Colvec) {
	numData := len(input)

	for layer := 1; layer <= len(neuNet.diffW)-1; layer++ {
		for j := range neuNet.diffW[layer] {
			for i := range neuNet.diffW[layer][j] {
				neuNet.diffW[layer][j][i] = 0.0
			}
		}
	}

	for data := 0; data <= numData-1; data++ {
		neuNet.Forward(input[data])

		layer := len(neuNet.W) - 1
		for j := range neuNet.Output[layer] {
			neuNet.Delta[layer][j] = neuNet.Output[layer][j] - correct[data][j]
		}
		for layer--; layer >= 1; layer-- {
			actFuncDiff := neuNet.ActFuncHidden[layer].Backward(neuNet.Midval[layer], neuNet.Output[layer])
			for j := range neuNet.Delta[layer] {
				neuNet.Delta[layer][j] = 0.0
				for k := range neuNet.Delta[layer+1] {
					neuNet.Delta[layer][j] += neuNet.Delta[layer+1][k] * neuNet.W[layer+1][k][j] * actFuncDiff[j]
				}
			}
		}

		for layer = 1; layer <= len(neuNet.W)-1; layer++ {
			for j := range neuNet.W[layer] {
				for i := range neuNet.W[layer][j] {
					neuNet.diffW[layer][j][i] += neuNet.Delta[layer][j] * neuNet.Output[layer-1][i]
				}
			}
		}
	}

	for layer := 1; layer <= len(neuNet.diffW)-1; layer++ {
		for j := range neuNet.diffW[layer] {
			for i := range neuNet.diffW[layer][j] {
				neuNet.diffW[layer][j][i] /= float64(numData)
			}
		}
	}
}

//GradDescent optimizes the neural network to fit the given dataset.
func (neuNet *NeuralNet) GradDescent(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			for i := range neuNet.W[layer][j] {
				neuNet.W[layer][j][i] -= neuNet.diffW[layer][j][i] * neuNet.ParamGradDecent.LearnRate
			}
		}
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//AdaGrad performs AdaGrad optimization.
func (neuNet *NeuralNet) AdaGrad(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	if neuNet.ParamAdaGrad.rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLeaRat := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = neuNet.ParamAdaGrad.LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLeaRat
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.sqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.sqSum[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaGrad.sqSum[layer][j])-1; i++ {
					neuNet.ParamAdaGrad.sqSum[layer][j][i] = neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdaGrad.rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.sqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.sqSum[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaGrad.sqSum[layer][j])-1; i++ {
					neuNet.ParamAdaGrad.sqSum[layer][j][i] += neuNet.diffW[layer][j][i] * neuNet.diffW[layer][j][i]
					neuNet.dW[layer][j][i] = -1.0 * neuNet.ParamAdaGrad.LearnRate * neuNet.diffW[layer][j][i] / math.Sqrt(neuNet.ParamAdaGrad.sqSum[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaGrad.rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//Momentum performs single run of momentum based optimization by using a (mini-) batch.
func (neuNet *NeuralNet) Momentum(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			for i := range neuNet.W[layer][j] {
				neuNet.dW[layer][j][i] = neuNet.ParamMomentum.MomentRate*neuNet.ParamMomentum.moment[layer][j][i] - (1.0-neuNet.ParamMomentum.MomentRate)*neuNet.ParamMomentum.LearnRate*neuNet.diffW[layer][j][i]
				neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
				neuNet.ParamMomentum.moment[layer][j][i] = neuNet.dW[layer][j][i]
			}
		}
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}

	err /= float64(numData)

	return
}

//RMSProp performs optimization of the nueralnet neuNet by using gradient descent with exponentially-decaying scaling factor.
func (neuNet *NeuralNet) RMSProp(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	DecayRate := neuNet.ParamRMSProp.DecayRate
	LearnRate := neuNet.ParamRMSProp.LearnRate

	if neuNet.ParamRMSProp.rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamRMSProp.expMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.expMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.expMvAv[layer][j])-1; i++ {
					neuNet.ParamRMSProp.expMvAv[layer][j][i] = (1-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamRMSProp.rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamRMSProp.expMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.expMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.expMvAv[layer][j])-1; i++ {
					neuNet.ParamRMSProp.expMvAv[layer][j][i] = DecayRate*neuNet.ParamRMSProp.expMvAv[layer][j][i] + (1-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i]
					neuNet.dW[layer][j][i] = -1.0 * LearnRate * neuNet.diffW[layer][j][i] / math.Sqrt(neuNet.ParamRMSProp.expMvAv[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamRMSProp.rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}
