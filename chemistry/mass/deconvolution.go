package mass

import (
	"PainTheMaster/mybraly/order"
	"PainTheMaster/mybraly/stat"
	"math"
)

//DeconvolutionParams is a bundle of parameters for deconvolution
type DeconvolutionParams struct {
	ClustersMax     int
	ChargeMin       int
	ChargeMax       int
	ClusterPeaksMin int
	AdditionalMass  float64
	MPerZErrMax     float64
	IgnoreLevel     float64
}

//Clusterizer clusters peaks from pks
func Clusterizer(pks Peaks, params DeconvolutionParams) {

	trimed := pks.noizeCut(params.IgnoreLevel)
	trimed.SortByMPerZ()

	//multiCharge corresponds to signals corrected by charge and cation mass
	multiCharge := make([]Peaks, params.ChargeMax-params.ChargeMin+1)
	for i := range multiCharge {
		multiCharge[i] = make(Peaks, trimed.Length())
		for j := range multiCharge[i] {
			multiCharge[i][j] = trimed[j]
			multiCharge[i][j].MPerZ = (trimed[j].MPerZ - params.AdditionalMass) * float64(params.ChargeMin+i)
		}
	}

	trimed.SortByIntensReverse()
	unchekced := 0
	//TODO: Please write clusterizer
	for numClust := 0; numClust <= params.ClustersMax-1; numClust++ {
		for ; trimed[unchekced].Cluster != ClusterUnAssigned; unchekced++ {
		}

		//		var id int
		//		for strongPeak
	}
}

func (pks Peaks) noizeCut(noizeLevel float64) (trimmed Peaks) {
	for i := range pks {
		if pks[i].Intens >= noizeLevel {
			trimmed = append(trimmed, pks[i])
		}
	}
	return
}

//fit calculates fitting with a comb.
//Sort the pks before use in m/z order
func (pks Peaks) fit(targetID int, interval int, additionalMass float64, backward int, forward int, maxError float64) (accumulation float64, clusterPksID order.SortableInt) {
	idSeq := pks.IDConversion()
	targetIndex := idSeq[targetID]

	clusterPksID = make(order.SortableInt, 1)
	clusterPksID[0] = targetID

	targetMPerZ := pks[targetIndex].MPerZ
	comb := make([]float64, forward+backward+1)
	for i := 0; i <= forward+backward; i++ {
		comb[i] = targetMPerZ + additionalMass*float64(i-backward)
	}

	accumulation = 0.0
	for i := 0; i <= forward+backward; i++ {
		match := pks.BinarySearchMPerZ(comb[i], maxError)
		match.SortByIntensReverse()

		for j := range match {
			if match[j].Cluster == ClusterUnAssigned {
				accumulation += match[j].Intens
				clusterPksID = append(clusterPksID, match[i].ID)
				break
			}
		}
	}

	order.QuickSort(clusterPksID)

	return
}

func (pks Peaks) optim(targetMPerZ float64, targetID int, clusterPksID order.SortableInt, interval int, additionalMass float64) (optimTargetMPerZ float64, stdev float64) {

	idSeq := pks.IDConversion()

	order.QuickSort(clusterPksID)

	customCompare := func(srt order.Sorter, toBeComp int) (result int) {
		asserted := srt.(order.SortableInt)
		if targetID < asserted[toBeComp] {
			result = -1
		} else if targetID > asserted[toBeComp] {
			result = 1
		} else {
			result = 0
		}
		return
	}

	targetIdx := order.BinarySearch(clusterPksID, customCompare)

	makeComb := func(targetMPerZ float64, backward int, forward int) (comb []float64) {
		comb = make([]float64, backward+forward+1)
		for i := 0; i <= backward+forward; i++ {
			comb[i] = targetMPerZ + additionalMass*float64(i-backward)
		}
		return
	}

	fitness := func(comb []float64) (accumError float64) {
		accumError = 0.0
		for i := 0; i <= clusterPksID.Length()-1; i++ {
			accumError += math.Abs(comb[i] - pks[idSeq[i]].MPerZ)
		}
		return
	}

	const (
		shiftRatio   = 0.001
		correctRatio = 0.1
		repetition   = 10
	)

	backward := targetIdx
	forward := (clusterPksID.Length() - 1) - targetIdx

	for cycle := 1; cycle <= repetition; cycle++ {
		comb := makeComb(targetMPerZ, backward, forward)
		shiftedMPerZ := targetMPerZ * (1.0 + shiftRatio)
		shiftedComb := makeComb(shiftedMPerZ, backward, forward)

		accumError := fitness(comb)
		shiftedAccumError := fitness(shiftedComb)

		differential := (shiftedAccumError - accumError) / (targetMPerZ * shiftRatio)
		targetMPerZ -= correctRatio * differential
	}

	optimTargetMPerZ = targetMPerZ
	comb := makeComb(optimTargetMPerZ, backward, forward)

	stdev = 0.0
	for i := 0; i <= clusterPksID.Length()-1; i++ {
		stdev += math.Pow(comb[i]-pks[idSeq[i]].MPerZ, 2.0)
	}
	stdev = math.Sqrt(stdev / float64(clusterPksID.Length()))

	return
}

//expand expands a cluster. pks has to be sorted ID-wise prior to use
func (pks Peaks) expand(targetMPerZ float64, targetID int, clusterPksID order.SortableInt, backward int, forward int, interval int,
	decParam DeconvolutionParams) (retTargetMPerZ float64, retClusterPksID order.SortableInt, retBackward int, retForward int) {

	const errPctRange = 90

	pks.SortByID()
	//	idSeq := pks.IDConversion()

	order.QuickSort(clusterPksID)

	deltaMPerZ := decParam.AdditionalMass * float64(interval)

	backID := clusterPksID[0]
	for backID != SearchNotFound {
		//TODO: use target m/z insted of target ID
		match := pks.BinarySearchMPerZ(targetMPerZ-deltaMPerZ*float64(backward+1), decParam.MPerZErrMax)
		if match == nil {
			backID = SearchNotFound
		} else {
			match.SortByIntensReverse()
			if match[0].Intens >= decParam.IgnoreLevel {
				backID = match[0].ID
				//				backMPerZ = pks[idSeq[backID]].MPerZ
				clusterPksID = append(clusterPksID, backID)
				backward++

				var stdev float64
				targetMPerZ, stdev = pks.optim(targetMPerZ, targetID, clusterPksID, interval, decParam.AdditionalMass)
				newErr := stdev * stat.TwoTailSigma(errPctRange)
				if newErr < decParam.MPerZErrMax {
					decParam.MPerZErrMax = newErr
				}
			} else {
				backID = SearchNotFound
			}
		}
	}

	order.QuickSort(clusterPksID)

	forwardID := clusterPksID[clusterPksID.Length()-1]
	for forwardID != SearchNotFound {
		//TODO: use target m/z insted of target ID
		match := pks.BinarySearchMPerZ(targetMPerZ+deltaMPerZ*float64(forward+1), decParam.MPerZErrMax)
		if match == nil {
			forwardID = SearchNotFound
		} else {
			match.SortByIntensReverse()
			if match[0].Intens >= decParam.IgnoreLevel {
				forwardID = match[0].ID
				//				forwardMPerZ = pks[idSeq[forwardID]].MPerZ
				clusterPksID = append(clusterPksID, forwardID)
				forward++

				var stdev float64
				targetMPerZ, stdev = pks.optim(targetMPerZ, targetID, clusterPksID, interval, decParam.AdditionalMass)
				newErr := stdev * stat.TwoTailSigma(errPctRange)
				if newErr < decParam.MPerZErrMax {
					decParam.MPerZErrMax = newErr
				}
			} else {
				forwardID = SearchNotFound
			}
		}
	}

	retTargetMPerZ = targetMPerZ
	retClusterPksID = clusterPksID
	retBackward = backward
	retForward = forward

	return
}
