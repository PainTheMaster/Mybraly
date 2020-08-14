package deeplearning

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"math"
)

//Labels for optimizers
const (
	LabelGradDec  = "GradDecent"
	LabelAdaGrad  = "AdaGrad"
	LabelMomentum = "Momentum"
	LabelRMSProp  = "RMSProp"
	LabelAdaDelta = "AdaDelta"
	LabelAdam     = "Adam"
)

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
				neuNet.dW[layer][j][i] = (-1.0) * neuNet.diffW[layer][j][i] * neuNet.ParamGradDecent.LearnRate
				neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
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

//AdaDelta performs optimization of the neuralnet with dimension consistent way.
func (neuNet *NeuralNet) AdaDelta(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	DecayRate := neuNet.ParamAdaDelta.DecayRate
	const LearnRateGradDec = 0.01

	if neuNet.ParamAdaDelta.rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRateGradDec
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.expMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.expMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.expMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] = (1.0-DecayRate)*neuNet.dW[layer][j][i]*neuNet.dW[layer][j][i] + smallNum
					neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] = (1.0-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdaDelta.rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.expMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.expMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.expMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] + (1-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i]
					neuNet.dW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.expMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i]) * neuNet.diffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
					neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.dW[layer][j][i]*neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}

//AltAdaDelta performs optimization of the neuralnet with dimension consistent way.
func (neuNet *NeuralNet) AltAdaDelta(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	DecayRate := neuNet.ParamAdaDelta.DecayRate

	if neuNet.ParamAdaDelta.rep == 0 {
		smallNum := 1.0e-8
		neuNet.Differentiate(input, correct)

		for layer := 1; layer <= len(neuNet.ParamAdaDelta.expMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.expMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.expMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] = smallNum
					neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] = (1.0-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i] + smallNum
					neuNet.dW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.expMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i]) * neuNet.diffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
					neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.dW[layer][j][i]*neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.expMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.expMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.expMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i] + (1-DecayRate)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i]
					neuNet.dW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.expMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.expMvAvGrad[layer][j][i]) * neuNet.diffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
					neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.expMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.dW[layer][j][i]*neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}

//Adam performs optimization of the nueralnet neuNet by using gradient descent with exponentially-decaying scaling factor.
func (neuNet *NeuralNet) Adam(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	LearnRate := neuNet.ParamAdam.LearnRate
	DecayRate1 := neuNet.ParamAdam.DecayRate1
	DecayRate2 := neuNet.ParamAdam.DecayRate2

	if neuNet.ParamAdam.rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamAdam.expMvAvPri)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdam.expMvAvPri[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdam.expMvAvPri[layer][j])-1; i++ {
					neuNet.ParamAdam.expMvAvPri[layer][j][i] = (1 - DecayRate1) * neuNet.diffW[layer][j][i]
					neuNet.ParamAdam.expMvAvSec[layer][j][i] = (1-DecayRate2)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdam.rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamRMSProp.expMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.expMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.expMvAv[layer][j])-1; i++ {
					neuNet.ParamAdam.expMvAvPri[layer][j][i] = DecayRate1*neuNet.ParamAdam.expMvAvPri[layer][j][i] + (1-DecayRate1)*neuNet.diffW[layer][j][i]
					neuNet.ParamAdam.expMvAvSec[layer][j][i] = DecayRate2*neuNet.ParamAdam.expMvAvSec[layer][j][i] + (1-DecayRate2)*neuNet.diffW[layer][j][i]*neuNet.diffW[layer][j][i]
					rep := neuNet.ParamAdam.rep
					neuNet.dW[layer][j][i] = -1.0 * LearnRate * neuNet.ParamAdam.expMvAvPri[layer][j][i] / (1 - math.Pow(DecayRate1, float64(rep))) /
						math.Sqrt(neuNet.ParamAdam.expMvAvSec[layer][j][i]/(1-math.Pow(DecayRate2, float64(rep))))
					neuNet.W[layer][j][i] += neuNet.dW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdam.rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}
