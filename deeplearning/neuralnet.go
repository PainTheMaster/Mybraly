package deeplearning

import (
	"PainTheMaster/mybraly/deeplearning/mnist"
	"PainTheMaster/mybraly/mymath/linearalgebra"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
)

//NeuralNet is a neural network composed of a weight matrix "W", bias vector "B",
//activation function for hdden layers "ActivFuncHidden", activation function for out-put layer "ActivFuncOut".
type NeuralNet struct {
	//Basic Structure parameters. These parameters are needed to store and restore the neuralnetwork
	Nodes            []int
	StrActFuncHidden []string
	StrActFuncOut    string

	//Weight and intermediate values for optimization.
	W     [][][]float64          //Weight
	Delta []linearalgebra.Colvec //Delta needed to calucate differential in back propagation
	DiffW [][][]float64          //differential of error in terms of W
	DW    [][][]float64          //correction of W

	//Drop out related parameters and intermediate values.
	DropRatio [][]float64            //The ratio in which a node of the layer is dropped.
	DropFlag  [][]bool               //True=> drop that cell. False=> don't drop it.
	NumDrop   [][]int                //the number of dropping that cell experienced.
	NumTrial  int                    //Total number of trial with drop out.
	Mask      []linearalgebra.Colvec //For two purpose

	//Weight decay related parameters
	WeightDecayCoeff float64

	//Activation functions
	ActFuncHidden []ActFuncHiddenSet
	ActivFuncOut  ActFuncOutputSet

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
		Rep   int
		SqSum [][][]float64
	}

	ParamRMSProp struct {
		//Hyper parameters
		LearnRate float64
		DecayRate float64
		//Working parameters
		Rep     int
		ExpMvAv [][][]float64
	}

	ParamAdaDelta struct {
		//Hyper parameters
		DecayRate float64
		//WorkingParameters
		Rep         int
		ExpMvAvDW   [][][]float64
		ExpMvAvGrad [][][]float64
	}

	ParamAdam struct {
		//Hyper parameters
		LearnRate  float64
		DecayRate1 float64
		DecayRate2 float64
		//Working parameters
		Rep        int
		ExpMvAvPri [][][]float64
		ExpMvAvSec [][][]float64
	}

	//Output and values needed to calculate it.
	Midval []linearalgebra.Colvec //u: values to be processed by activation function
	Output []linearalgebra.Colvec //z: values processed by activation function
}

//Make makes a new empty nerral network "neuralNet". "nodes" represents the number of nodes in each layer
func Make(nodes []int, strActFuncHidden []string, strActFuncOut string) (neuralNet NeuralNet) {
	layers := len(nodes)

	neuralNet.W = make([][][]float64, layers)
	neuralNet.DW = make([][][]float64, layers)
	neuralNet.DiffW = make([][][]float64, layers)
	neuralNet.DropRatio = make([][]float64, layers)
	neuralNet.DropFlag = make([][]bool, layers)
	neuralNet.NumDrop = make([][]int, layers)
	neuralNet.Mask = make([]linearalgebra.Colvec, layers)
	neuralNet.ParamMomentum.moment = make([][][]float64, layers)
	neuralNet.ParamAdaGrad.SqSum = make([][][]float64, layers)
	neuralNet.ParamRMSProp.ExpMvAv = make([][][]float64, layers)
	neuralNet.ParamAdaDelta.ExpMvAvDW = make([][][]float64, layers)
	neuralNet.ParamAdaDelta.ExpMvAvGrad = make([][][]float64, layers)
	neuralNet.ParamAdam.ExpMvAvPri = make([][][]float64, layers)
	neuralNet.ParamAdam.ExpMvAvSec = make([][][]float64, layers)

	neuralNet.Midval = make([]linearalgebra.Colvec, layers)
	neuralNet.Output = make([]linearalgebra.Colvec, layers)

	neuralNet.Mask[0] = make(linearalgebra.Colvec, nodes[0])

	for i := 1; i <= layers-1; i++ {
		neuralNet.W[i] = make([][]float64, nodes[i])
		neuralNet.DW[i] = make([][]float64, nodes[i])
		neuralNet.DiffW[i] = make([][]float64, nodes[i])
		neuralNet.DropRatio[i] = make([]float64, nodes[i])
		neuralNet.DropFlag[i] = make([]bool, nodes[i])
		neuralNet.NumDrop[i] = make([]int, nodes[i])
		neuralNet.Mask[i] = make(linearalgebra.Colvec, nodes[i])
		neuralNet.ParamMomentum.moment[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaGrad.SqSum[i] = make([][]float64, nodes[i])
		neuralNet.ParamRMSProp.ExpMvAv[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaDelta.ExpMvAvDW[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdaDelta.ExpMvAvGrad[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdam.ExpMvAvPri[i] = make([][]float64, nodes[i])
		neuralNet.ParamAdam.ExpMvAvSec[i] = make([][]float64, nodes[i])
		for j := range neuralNet.W[i] {
			neuralNet.W[i][j] = make([]float64, nodes[i-1]+1) //The last column (nodes[i-1]-th) is bias
			neuralNet.DW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.DiffW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamMomentum.moment[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaGrad.SqSum[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamRMSProp.ExpMvAv[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaDelta.ExpMvAvDW[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdaDelta.ExpMvAvGrad[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdam.ExpMvAvPri[i][j] = make([]float64, nodes[i-1]+1)
			neuralNet.ParamAdam.ExpMvAvSec[i][j] = make([]float64, nodes[i-1]+1)
		}

		neuralNet.Midval[i] = make(linearalgebra.Colvec, nodes[i])
		neuralNet.Output[i] = make(linearalgebra.Colvec, nodes[i]+1) //The last elemenn is for the bias in the next layer
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

	var seed int64 = 100
	randSource := rand.NewSource(seed)
	newRand := rand.New(randSource)
	randAve := 0.00
	smallPosBias := 0.01
	for l := 1; l <= layers-2; l++ {
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l][0])
			randStdev := neuralNet.ActFuncHidden[l].StdevWtFunc(numBranch)
			br := 0
			neuralNet.W[l][node][br] = smallPosBias
			for br = 1; br <= len(neuralNet.W[l][node])-1; br++ {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randAve
			}
		}
	}
	{
		l := layers - 1
		for node := range neuralNet.W[l] {
			numBranch := len(neuralNet.W[l][0])
			randStdev := neuralNet.ActivFuncOut.StdevWtFunc(numBranch)
			br := 0
			neuralNet.W[l][node][br] = smallPosBias
			for br = 1; br <= len(neuralNet.W[l][node])-1; br++ {
				neuralNet.W[l][node][br] = newRand.NormFloat64()*randStdev + randAve
			}
		}
	}

	neuralNet.ParamAdaGrad.Rep = 0

	for layer := range neuralNet.Mask {
		for j := range neuralNet.Mask[layer] {
			neuralNet.Mask[layer][j] = 1.0
		}
	}

	neuralNet.Nodes = nodes
	neuralNet.StrActFuncHidden = strActFuncHidden
	neuralNet.StrActFuncOut = strActFuncOut

	return
}

//Forward calculates output of a neural "neuralNet" from the input "input".
func (neuNet *NeuralNet) Forward(input linearalgebra.Colvec) {
	if len(neuNet.W[1][0]) != len(input)+1 {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	for i := 0; i <= len(neuNet.Mask[0])-2; i++ {
		neuNet.Output[0][i] = input[i] * neuNet.Mask[0][i]
	}

	for layer := 1; layer <= len(neuNet.W)-2; layer++ {
		neuNet.Midval[layer] = linearalgebra.MatColvecMult(neuNet.W[layer], neuNet.Output[layer-1])

		neuNet.Output[layer] = neuNet.ActFuncHidden[layer].Forward(neuNet.Midval[layer])
		for i := range neuNet.Output[layer] {
			neuNet.Output[layer][i] *= neuNet.Mask[layer][i]
		}
		neuNet.Output[layer] = append(neuNet.Output[layer], 1.0)
	}

	{
		layer := len(neuNet.W) - 1
		neuNet.Midval[layer] = linearalgebra.MatColvecMult(neuNet.W[layer], neuNet.Output[layer-1])
		neuNet.Output[layer] = neuNet.ActivFuncOut.Forward(neuNet.Midval[layer])
	}

	return
}

//ForwardDrop performs forward dropping with reduced calculation load.
func (neuNet *NeuralNet) ForwardDrop(input linearalgebra.Colvec) {
	if len(neuNet.W[1][0]) != len(input)+1 {
		fmt.Println("deeplearing.Forward() error: input vector mismatch.")
	}

	//	for i := 0; i <= len(neuNet.Mask[0])-2; i++ {
	for i := 0; i <= neuNet.Nodes[0]-1; i++ {
		if neuNet.DropFlag[0][i] {
			neuNet.Output[0][i] = 0.0
		} else {
			neuNet.Output[0][i] = input[i]
		}
	}

	for layer := 1; layer <= len(neuNet.W)-2; layer++ {
		neuNet.Midval[layer] = make(linearalgebra.Colvec, neuNet.Nodes[layer])
		for i := 0; i <= neuNet.Nodes[layer]-1; i++ {
			if neuNet.DropFlag[layer][i] {
				//				neuNet.Midval[layer][i] = 0.0	 Just to show the intention. The mid value is useless in this case.
			} else {
				neuNet.Midval[layer][i] = helperVecInnerProd(neuNet.W[layer][i], neuNet.Output[layer-1])
			}
		}
		neuNet.Output[layer] = neuNet.ActFuncHidden[layer].Forward(neuNet.Midval[layer])

		for i := range neuNet.Output[layer] {
			if neuNet.DropFlag[layer][i] {
				neuNet.Output[layer][i] = 0.0
			}
		}
		neuNet.Output[layer] = append(neuNet.Output[layer], 1.0)
	}

	{
		layer := len(neuNet.W) - 1
		neuNet.Midval[layer] = linearalgebra.MatColvecMult(neuNet.W[layer], neuNet.Output[layer-1])
		neuNet.Output[layer] = neuNet.ActivFuncOut.Forward(neuNet.Midval[layer])
	}

	return
}

func helperVecInnerProd(v1, v2 []float64) (ans float64) {
	if len(v1) != len(v2) {
		fmt.Println("helperVecInnerProd error: v1, v2 dimension mismatch")
	}
	for i := range v1 {
		ans += v1[i] * v2[i]
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

/////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// General purpose functions /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

//GeneralTraining is a function for traiing with general data form.
func (neuNet *NeuralNet) GeneralTraining(trainInput, trainOutput []linearalgebra.Colvec, sizeMiniBatch, repet int, labelOptim string) (errHist []float64) {
	var optimizer func(input, correct []linearalgebra.Colvec) (err float64)
	switch labelOptim {
	case LabelGradDesc:
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
	case LabelGradDescWD:
		optimizer = neuNet.GradDescentWeightDecay
	case LabelAdaGradWD:
		optimizer = neuNet.AdaGradWeightDecay
	case LabelMomentumWD:
		optimizer = neuNet.MomentumWeightDecay
	case LabelRMSPropWD:
		optimizer = neuNet.RMSPropWeightDecay
	case LabelAdaDeltaWD:
		optimizer = neuNet.AdaDeltaWeightDecay
	}

	numData := len(trainInput)

	randSeed := int64(100)
	randSource := rand.NewSource(randSeed)
	randGener := rand.New(randSource)

	for try := 1; try <= repet; try++ {
		var miniBatchInput, miniBatchOutput []linearalgebra.Colvec
		picked := make([]bool, numData)
		for sample := 0; sample <= sizeMiniBatch-1; sample++ {
			for {
				id := randGener.Intn(numData) //this returns a randdum int from 0 to (numImgs-1)
				if !picked[id] {
					picked[id] = true
					miniBatchInput = append(miniBatchInput, trainInput[id])
					miniBatchOutput = append(miniBatchOutput, trainOutput[id])
					break
				}
			}
		}
		tempErr := optimizer(miniBatchInput, miniBatchOutput)
		errHist = append(errHist, tempErr)
	}
	return
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////// MNIST related functions /////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

//MnistTrain trains the neural network.
func (neuNet *NeuralNet) MnistTrain(trainImg, trainLabel *os.File, sizeMiniBatch, repet int, labelOptim string) (errHist []float64) {
	var optimizer func(input, correct []linearalgebra.Colvec) (err float64)
	switch labelOptim {
	case LabelGradDesc:
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
	case LabelGradDescWD:
		optimizer = neuNet.GradDescentWeightDecay
	case LabelAdaGradWD:
		optimizer = neuNet.AdaGradWeightDecay
	case LabelMomentumWD:
		optimizer = neuNet.MomentumWeightDecay
	case LabelRMSPropWD:
		optimizer = neuNet.RMSPropWeightDecay
	case LabelAdaDeltaWD:
		optimizer = neuNet.AdaDeltaWeightDecay
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
		//		fmt.Printf("%d-th run ended. Error=%f\n", try, tempErr)
		errHist = append(errHist, tempErr)
	}
	return

}

//MnistTest performs test
func (neuNet NeuralNet) MnistTest(testImg, testLabel *os.File, repet int) (accuracyPct float64, errdata [][]int) {

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
		var id int
		for {
			id = randGener.Intn(numImgs)
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
			errdata = append(errdata, []int{id, pickedLabel, infLabel})
			//			fmt.Printf("Not OK: %d, corr:%d, inf: %d\n", id, pickedLabel, infLabel)
		}
	}

	accuracyPct = float64(ok) / float64(ok+nok) * 100.0

	return

}

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////    Store and restore    //////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
type storeStruc struct {
	W                [][][]float64
	DW               [][][]float64
	DiffW            [][][]float64
	DropRatio        []linearalgebra.Colvec
	Nodes            []int
	StrActFuncHidden []string
	StrActFuncOut    string
	ParamGradDecent  struct{ LearnRate float64 }
	ParamMomentum    struct {
		LearnRate  float64
		MomentRate float64
		moment     [][][]float64
	}
	ParamAdaGrad struct {
		LearnRate float64
		Rep       int
		SqSum     [][][]float64
	}
	ParamRMSProp struct {
		LearnRate float64
		DecayRate float64
		Rep       int
		ExpMvAv   [][][]float64
	}

	ParamAdaDelta struct {
		DecayRate   float64
		Rep         int
		ExpMvAvDW   [][][]float64
		ExpMvAvGrad [][][]float64
	}
	ParamAdam struct {
		LearnRate  float64
		DecayRate1 float64
		DecayRate2 float64
		Rep        int
		ExpMvAvPri [][][]float64
		ExpMvAvSec [][][]float64
	}
	WeightDecayCoeff float64
}

//Store stores current neuralnetwork to a file designated by the fileName
func (neuNet NeuralNet) Store(fileName string) {

	ss := storeStruc{
		W:                neuNet.W,
		DW:               neuNet.DW,
		DiffW:            neuNet.DiffW,
		DropRatio:        neuNet.Mask,
		Nodes:            neuNet.Nodes,
		StrActFuncHidden: neuNet.StrActFuncHidden,
		StrActFuncOut:    neuNet.StrActFuncOut,
		ParamAdaGrad:     neuNet.ParamAdaGrad,
		ParamRMSProp:     neuNet.ParamRMSProp,
		ParamAdaDelta:    neuNet.ParamAdaDelta,
		ParamAdam:        neuNet.ParamAdam,
		WeightDecayCoeff: neuNet.WeightDecayCoeff,
	}

	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println(err)
	}

	gobEncoder := gob.NewEncoder(file)
	err = gobEncoder.Encode(ss)
	if err != nil {
		fmt.Println(err)
	}
	file.Close()
}

//Restore restoress a neuralnetwork from a designated file
func Restore(fileName string) (neuNet NeuralNet) {
	var ss storeStruc

	file, err := os.Open(fileName)
	if err != nil {
		fmt.Println(err)
	}

	gobDecoder := gob.NewDecoder(file)
	err = gobDecoder.Decode(ss)
	if err != nil {
		fmt.Println(err)
	}

	neuNet = Make(ss.Nodes, ss.StrActFuncHidden, ss.StrActFuncOut)
	neuNet.W = ss.W
	neuNet.DW = ss.DW
	neuNet.DiffW = ss.DiffW
	neuNet.Mask = ss.DropRatio
	neuNet.Nodes = ss.Nodes
	neuNet.ParamAdaGrad = ss.ParamAdaGrad
	neuNet.ParamRMSProp = ss.ParamRMSProp
	neuNet.ParamAdaDelta = ss.ParamAdaDelta
	neuNet.ParamAdam = ss.ParamAdam
	neuNet.WeightDecayCoeff = ss.WeightDecayCoeff

	return
}
