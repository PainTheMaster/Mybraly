package deeplearning

import (
	"PainTheMaster/mybraly/deeplearning/mnist"
	"PainTheMaster/mybraly/math/linearalgebra"
	"fmt"
	"math"
	"math/rand"
	"os"
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

	ParamAdaDelta struct {
		//Hyper parameters
		DecayRate float64
		//WorkingParameters
		rep         int
		expMvAvDW   [][][]float64
		expMvAvGrad [][][]float64
	}

	ParamAdam struct {
		//Hyper parameters
		LearnRate  float64
		DecayRate1 float64
		DecayRate2 float64
		//Working parameters
		rep        int
		expMvAvPri [][][]float64
		expMvAvSec [][][]float64
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
	neuralNet.ParamAdaDelta.expMvAvDW = make([][][]float64, layers)
	neuralNet.ParamAdaDelta.expMvAvGrad = make([][][]float64, layers)
	neuralNet.ParamAdam.expMvAvPri = make([][][]float64, layers)
	neuralNet.ParamAdam.expMvAvSec = make([][][]float64, layers)

	neuralNet.Midval = make([]linearalgebra.Colvec, layers)
	neuralNet.Output = make([]linearalgebra.Colvec, layers)

	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		neuralNet.dW[i] = make([][]float64, nodes[i])
		neuralNet.diffW[i] = make([][]float64, nodes[i])
		neuralNet.ParamMomentum.moment[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaGrad.sqSum[i] = make([][]float64, nodes[i])
		neuralNet.ParamRMSProp.expMvAv[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaDelta.expMvAvDW[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaDelta.expMvAvGrad[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdam.expMvAvPri[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdam.expMvAvSec[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1) //The last (nodes[i-1]-th) column is bias
			neuralNet.dW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.diffW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamMomentum.moment[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaGrad.sqSum[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamRMSProp.expMvAv[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaDelta.expMvAvDW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaDelta.expMvAvGrad[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdam.expMvAvPri[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdam.expMvAvSec[i][j] = make([]float64, nodes[i-1]+1)
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

//Train trains the neural network.
func (neuNet *NeuralNet) Train(trainImg, trainLabel *os.File, sizeMiniBatch, repet int, labelOptim string) (errHist []float64) {
	var optimizer func(input, correct []linearalgebra.Colvec) (err float64)
	switch labelOptim {
	case LabelGradDec:
		optimizer = neuNet.GradDescent
	case LabelAdaGrad:
		optimizer = neuNet.AdaGrad
	case LabelMomentum:
		optimizer = neuNet.Momentum
	case LabelRMSProp:
		optimizer = neuNet.RMSProp
	case LabelAdaDelta:
		optimizer = neuNet.AdaDelta
	case LabelAdam:
		optimizer = neuNet.Adam
	}

	trainInput := make([]linearalgebra.Colvec, sizeMiniBatch)
	trainLabelOneHot := make([]linearalgebra.Colvec, sizeMiniBatch)

	numImgs := mnist.GetNumItems(trainImg)

	randSeed := int64(100)
	randSource := rand.NewSource(randSeed)
	randGener := rand.New(randSource)

	for try := 1; try <= repet; try++ {
		picked := make([]bool, numImgs)
		for sample := 0; sample <= sizeMiniBatch-1; sample++ {
			for {
				id := randGener.Intn(numImgs) //this returns a randdum int from 0 to (numImgs-1)
				if !picked[id] {
					picked[id] = true
					trainInput[sample] = mnist.ImagToColvec(mnist.GetImage(trainImg, id))
					trainLabelOneHot[sample] = mnist.LabelOneHot(mnist.GetLabel(trainLabel, id))
					break
				}
			}
		}
		tempErr := optimizer(trainInput, trainLabelOneHot)
		fmt.Printf("%d-th run ended. Error=%f\n", try, tempErr)
		errHist = append(errHist, tempErr)
	}
	return

}

//Test performs test
func (neuNet NeuralNet) Test(testImg, testLabel *os.File, repet int) (accuracyPct float64) {

	layerOutput := len(neuNet.Output) - 1
	searchMax := func(vec linearalgebra.Colvec) (idxMax int) {
		idxMax = 0
		for i := range vec {
			if vec[i] > vec[idxMax] {
				idxMax = i
			}
		}
		return
	}

	numImgs := mnist.GetNumItems(testImg)

	randSeed := int64(100)
	randSource := rand.NewSource(randSeed)
	randGener := rand.New(randSource)

	var ok, nok int

	picked := make([]bool, numImgs)
	for try := 1; try <= repet; try++ {
		var pickedImg linearalgebra.Colvec
		var pickedLabel int
		for {
			id := randGener.Intn(numImgs)
			if !picked[id] {
				pickedImg = mnist.ImagToColvec(mnist.GetImage(testImg, id))
				pickedLabel = mnist.GetLabel(testLabel, id)
				picked[id] = true
				break
			}
		}
		neuNet.Forward(pickedImg)
		infer := neuNet.Output[layerOutput]
		infLabel := searchMax(infer)
		if infLabel == pickedLabel {
			ok++
		} else {
			nok++
		}
	}

	accuracyPct = float64(ok) / float64(ok+nok) * 100.0

	return

}
