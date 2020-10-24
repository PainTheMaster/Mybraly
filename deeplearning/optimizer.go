package deeplearning

import (
	"PainTheMaster/mybraly/mymath/linearalgebra"
	"math"
)

//Labels for optimizers
const (
	LabelGradDesc   = "GradDesc"
	LabelAdaGrad    = "AdaGrad"
	LabelMomentum   = "Momentum"
	LabelRMSProp    = "RMSProp"
	LabelAdaDelta   = "AdaDelta"
	LabelAdam       = "Adam"
	LabelGradDescWD = "GradDesctWD"
	LabelAdaGradWD  = "AdaGradWD"
	LabelMomentumWD = "MomentumWD"
	LabelRMSPropWD  = "RMSPropWD"
	LabelAdaDeltaWD = "AdaDeltaWD"
)

//Differentiate calculates differentiation of error function dE/dw of the neuralnetwork based on a given dataset input and correct.
//The out put is averaged.
func (neuNet *NeuralNet) Differentiate(input, correct []linearalgebra.Colvec) {
	numData := len(input)

	//Resettin the Diff matrix
	for layer := 1; layer <= len(neuNet.DiffW)-1; layer++ {
		for j := range neuNet.DiffW[layer] {
			for i := range neuNet.DiffW[layer][j] {
				neuNet.DiffW[layer][j][i] = 0.0
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

				//calculate the Delta only if the DropFlag is not set
				if !neuNet.DropFlag[layer][j] {
					for k := range neuNet.Delta[layer+1] {
						//Delta[layer+1][k] == 0, so this calcuatation automaticaly takes the dropping out of the next layer into account
						neuNet.Delta[layer][j] += neuNet.Delta[layer+1][k] * neuNet.W[layer+1][k][j] * actFuncDiff[j]
					}
				}
			}
		}

		//TODO: the nest is too deep. Improve this somehow.
		for layer = 1; layer <= len(neuNet.W)-1; layer++ {
			for j := range neuNet.W[layer] {
				//Calculate the diffW only when DropFlag is not set. Otherwise, DiffW remain to be 0.
				if !neuNet.DropFlag[layer][j] {
					for i := range neuNet.W[layer][j] {
						//Add to diffW only when the node in the previous layer ([layer-1][i]) is alive.
						if !neuNet.DropFlag[layer-1][i] {
							neuNet.DiffW[layer][j][i] += neuNet.Delta[layer][j] * neuNet.Output[layer-1][i]
						}
					}
				}
			}
		}
	}

	reverseNumData := 1.0 / float64(numData)
	for layer := 1; layer <= len(neuNet.DiffW)-1; layer++ {
		for j := range neuNet.DiffW[layer] {
			for i := range neuNet.DiffW[layer][j] {
				neuNet.DiffW[layer][j][i] *= reverseNumData
			}
		}
	}
}

//GradDescent optimizes the neural network to fit the given dataset. Dropout compatible.
func (neuNet *NeuralNet) GradDescent(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			if !neuNet.DropFlag[layer][j] {
				for i := range neuNet.W[layer][j] {
					neuNet.DW[layer][j][i] = (-1.0) * neuNet.DiffW[layer][j][i] * neuNet.ParamGradDecent.LearnRate
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
			} else {
				for i := range neuNet.W[layer][j] {
					neuNet.DW[layer][j][i] = 0.0
				}
				neuNet.NumDrop[layer][j]++
			}
		}
	}

	neuNet.NumDropoutTrial++

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//GradDescentWeightDecay is gradient descent with weight decay.
func (neuNet *NeuralNet) GradDescentWeightDecay(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			for i := 0; i <= len(neuNet.W[layer][j])-2; i++ {
				neuNet.DW[layer][j][i] = (-1.0) * neuNet.ParamGradDecent.LearnRate * (neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i])
				neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
			}
			{
				i := len(neuNet.W[layer][j]) - 1
				neuNet.DW[layer][j][i] = (-1.0) * neuNet.DiffW[layer][j][i] * neuNet.ParamGradDecent.LearnRate
				neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
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

	if neuNet.ParamAdaGrad.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLeaRat := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = neuNet.ParamAdaGrad.LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLeaRat
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.SqSum[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaGrad.SqSum[layer][j])-1; i++ {
					neuNet.ParamAdaGrad.SqSum[layer][j][i] = neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.SqSum[layer])-1; j++ {
				if !neuNet.DropFlag[layer][j] {
					for i := 0; i <= len(neuNet.ParamAdaGrad.SqSum[layer][j])-1; i++ {
						neuNet.ParamAdaGrad.SqSum[layer][j][i] += neuNet.DiffW[layer][j][i] * neuNet.DiffW[layer][j][i]
						neuNet.DW[layer][j][i] = -1.0 * neuNet.ParamAdaGrad.LearnRate * neuNet.DiffW[layer][j][i] / math.Sqrt(neuNet.ParamAdaGrad.SqSum[layer][j][i])
						neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					}
				} else {
					neuNet.NumDrop[layer][j]++
				}
			}
		}

	}

	neuNet.NumDropoutTrial++
	neuNet.ParamAdaGrad.Rep++

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//AdaGradWeightDecay is AdaGrad with weight decay.
func (neuNet *NeuralNet) AdaGradWeightDecay(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	if neuNet.ParamAdaGrad.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLeaRat := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = neuNet.ParamAdaGrad.LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLeaRat
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.SqSum[layer])-2; j++ {
				for i := 0; i <= len(neuNet.ParamAdaGrad.SqSum[layer][j])-2; i++ {
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamAdaGrad.SqSum[layer][j][i] = diffModif*diffModif + smallNum
				}
				{
					i := len(neuNet.ParamAdaGrad.SqSum[layer][j]) - 1
					neuNet.ParamAdaGrad.SqSum[layer][j][i] = neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdaGrad.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaGrad.SqSum)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaGrad.SqSum[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaGrad.SqSum[layer][j])-2; i++ {
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamAdaGrad.SqSum[layer][j][i] += diffModif * diffModif
					neuNet.DW[layer][j][i] = -1.0 * neuNet.ParamAdaGrad.LearnRate * diffModif / math.Sqrt(neuNet.ParamAdaGrad.SqSum[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
				{
					i := len(neuNet.ParamAdaGrad.SqSum[layer][j]) - 1
					neuNet.ParamAdaGrad.SqSum[layer][j][i] += neuNet.DiffW[layer][j][i] * neuNet.DiffW[layer][j][i]
					neuNet.DW[layer][j][i] = -1.0 * neuNet.ParamAdaGrad.LearnRate * neuNet.DiffW[layer][j][i] / math.Sqrt(neuNet.ParamAdaGrad.SqSum[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaGrad.Rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)
	return
}

//Momentum performs single run of momentum based optimization by using a (mini-) batch. Dropout compatible.
func (neuNet *NeuralNet) Momentum(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			//If not dropped
			if !neuNet.DropFlag[layer][j] {
				for i := range neuNet.W[layer][j] {
					neuNet.DW[layer][j][i] = neuNet.ParamMomentum.MomentRatio*neuNet.ParamMomentum.moment[layer][j][i] - (1.0-neuNet.ParamMomentum.MomentRatio)*neuNet.ParamMomentum.LearnRate*neuNet.DiffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					neuNet.ParamMomentum.moment[layer][j][i] = neuNet.DW[layer][j][i]
				}
				//If dropped
			} else {
				for i := range neuNet.W[layer][j] {
					neuNet.DW[layer][j][i] = 0.0
					neuNet.ParamMomentum.moment[layer][j][i] = neuNet.DW[layer][j][i]
				}
				neuNet.NumDrop[layer][j]++
			}
		}
	}

	neuNet.NumDropoutTrial++

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}

	err /= float64(numData)

	return
}

//MomentumWeightDecay is momentum optimmizer with weight decay
func (neuNet *NeuralNet) MomentumWeightDecay(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)

	neuNet.Differentiate(input, correct)

	for layer := 1; layer <= len(neuNet.W)-1; layer++ {
		for j := range neuNet.W[layer] {
			for i := 0; i <= len(neuNet.W[layer][j])-2; i++ {
				diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
				neuNet.DW[layer][j][i] = neuNet.ParamMomentum.MomentRatio*neuNet.ParamMomentum.moment[layer][j][i] - (1.0-neuNet.ParamMomentum.MomentRatio)*neuNet.ParamMomentum.LearnRate*diffModif
				neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				neuNet.ParamMomentum.moment[layer][j][i] = neuNet.DW[layer][j][i]
			}
			{
				i := len(neuNet.W[layer][j]) - 1
				neuNet.DW[layer][j][i] = neuNet.ParamMomentum.MomentRatio*neuNet.ParamMomentum.moment[layer][j][i] - (1.0-neuNet.ParamMomentum.MomentRatio)*neuNet.ParamMomentum.LearnRate*neuNet.DiffW[layer][j][i]
				neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				neuNet.ParamMomentum.moment[layer][j][i] = neuNet.DW[layer][j][i]
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

	if neuNet.ParamRMSProp.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamRMSProp.ExpMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.ExpMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.ExpMvAv[layer][j])-1; i++ {
					neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamRMSProp.ExpMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.ExpMvAv[layer])-1; j++ {
				if !neuNet.DropFlag[layer][j] {
					for i := 0; i <= len(neuNet.ParamRMSProp.ExpMvAv[layer][j])-1; i++ {
						neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = DecayRate*neuNet.ParamRMSProp.ExpMvAv[layer][j][i] + (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i]
						neuNet.DW[layer][j][i] = -1.0 * LearnRate * neuNet.DiffW[layer][j][i] / math.Sqrt(neuNet.ParamRMSProp.ExpMvAv[layer][j][i])
						neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					}
				} else {
					neuNet.NumDrop[layer][j]++
				}
			}
		}

	}

	neuNet.ParamRMSProp.Rep++
	neuNet.NumDropoutTrial++

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}

//RMSPropWeightDecay is RMS Prop with weight decay
func (neuNet *NeuralNet) RMSPropWeightDecay(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	DecayRate := neuNet.ParamRMSProp.DecayRate
	LearnRate := neuNet.ParamRMSProp.LearnRate

	if neuNet.ParamRMSProp.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRate
		neuNet.GradDescentWeightDecay(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamRMSProp.ExpMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.ExpMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.ExpMvAv[layer][j])-2; i++ {
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = (1-DecayRate)*neuNet.DiffW[layer][j][i]*diffModif + smallNum
				}
				{
					i := len(neuNet.ParamRMSProp.ExpMvAv[layer][j]) - 1
					neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamRMSProp.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamRMSProp.ExpMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.ExpMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.ExpMvAv[layer][j])-2; i++ {
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = DecayRate*neuNet.ParamRMSProp.ExpMvAv[layer][j][i] + (1-DecayRate)*diffModif*diffModif
					neuNet.DW[layer][j][i] = -1.0 * LearnRate * diffModif / math.Sqrt(neuNet.ParamRMSProp.ExpMvAv[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
				{
					i := len(neuNet.ParamRMSProp.ExpMvAv[layer][j]) - 1
					neuNet.ParamRMSProp.ExpMvAv[layer][j][i] = DecayRate*neuNet.ParamRMSProp.ExpMvAv[layer][j][i] + (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i]
					neuNet.DW[layer][j][i] = -1.0 * LearnRate * neuNet.DiffW[layer][j][i] / math.Sqrt(neuNet.ParamRMSProp.ExpMvAv[layer][j][i])
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamRMSProp.Rep++
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

	if neuNet.ParamAdaDelta.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRateGradDec
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = (1.0-DecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i] + smallNum
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = (1.0-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] + (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i]
					neuNet.DW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i]) * neuNet.DiffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}

//AdaDeltaWeightDecay is Ada delta with weight decay.
func (neuNet *NeuralNet) AdaDeltaWeightDecay(input, correct []linearalgebra.Colvec) (err float64) {
	numData := len(input)
	AdaDelDecayRate := neuNet.ParamAdaDelta.DecayRate
	const LearnRateGradDec = 0.01

	if neuNet.ParamAdaDelta.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRateGradDec
		neuNet.GradDescentWeightDecay(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = (1.0-AdaDelDecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i] + smallNum
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = (1.0-AdaDelDecayRate)*diffModif*diffModif + smallNum
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					diffModif := neuNet.DiffW[layer][j][i] + neuNet.WeightDecayCoeff*neuNet.W[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = AdaDelDecayRate*neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] + (1-AdaDelDecayRate)*diffModif*diffModif
					neuNet.DW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i]) * diffModif
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = AdaDelDecayRate*neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] + (1-AdaDelDecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
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

	if neuNet.ParamAdaDelta.Rep == 0 {
		smallNum := 1.0e-8
		neuNet.Differentiate(input, correct)

		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = smallNum
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = (1.0-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
					neuNet.DW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i]) * neuNet.DiffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamAdaDelta.ExpMvAvDW)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j])-1; i++ {
					neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i] + (1-DecayRate)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i]
					neuNet.DW[layer][j][i] = -1.0 * math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i]) / math.Sqrt(neuNet.ParamAdaDelta.ExpMvAvGrad[layer][j][i]) * neuNet.DiffW[layer][j][i]
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
					neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] = DecayRate*neuNet.ParamAdaDelta.ExpMvAvDW[layer][j][i] + (1-DecayRate)*neuNet.DW[layer][j][i]*neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdaDelta.Rep++
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

	if neuNet.ParamAdam.Rep == 0 {
		smallNum := 1.0e-8
		tempGradDecLearnRate := neuNet.ParamGradDecent.LearnRate
		neuNet.ParamGradDecent.LearnRate = LearnRate
		neuNet.GradDescent(input, correct)
		neuNet.ParamGradDecent.LearnRate = tempGradDecLearnRate
		for layer := 1; layer <= len(neuNet.ParamAdam.ExpMvAvPri)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamAdam.ExpMvAvPri[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamAdam.ExpMvAvPri[layer][j])-1; i++ {
					neuNet.ParamAdam.ExpMvAvPri[layer][j][i] = (1 - DecayRate1) * neuNet.DiffW[layer][j][i]
					neuNet.ParamAdam.ExpMvAvSec[layer][j][i] = (1-DecayRate2)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i] + smallNum
				}
			}
		}
		neuNet.ParamAdam.Rep++
	} else {
		neuNet.Differentiate(input, correct)
		for layer := 1; layer <= len(neuNet.ParamRMSProp.ExpMvAv)-1; layer++ {
			for j := 0; j <= len(neuNet.ParamRMSProp.ExpMvAv[layer])-1; j++ {
				for i := 0; i <= len(neuNet.ParamRMSProp.ExpMvAv[layer][j])-1; i++ {
					neuNet.ParamAdam.ExpMvAvPri[layer][j][i] = DecayRate1*neuNet.ParamAdam.ExpMvAvPri[layer][j][i] + (1-DecayRate1)*neuNet.DiffW[layer][j][i]
					neuNet.ParamAdam.ExpMvAvSec[layer][j][i] = DecayRate2*neuNet.ParamAdam.ExpMvAvSec[layer][j][i] + (1-DecayRate2)*neuNet.DiffW[layer][j][i]*neuNet.DiffW[layer][j][i]
					rep := neuNet.ParamAdam.Rep
					neuNet.DW[layer][j][i] = -1.0 * LearnRate * neuNet.ParamAdam.ExpMvAvPri[layer][j][i] / (1 - math.Pow(DecayRate1, float64(rep))) /
						math.Sqrt(neuNet.ParamAdam.ExpMvAvSec[layer][j][i]/(1-math.Pow(DecayRate2, float64(rep))))
					neuNet.W[layer][j][i] += neuNet.DW[layer][j][i]
				}
			}
		}
		neuNet.ParamAdam.Rep++
	}

	err = 0.0
	for data := range input {
		err += neuNet.Error(input[data], correct[data])
	}
	err /= float64(numData)

	return
}
