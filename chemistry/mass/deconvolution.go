package mass

import (
	"PainTheMaster/mybraly/order"
	"PainTheMaster/mybraly/stat"
	"math"
)

const (
	ClusterNotFormed = -1
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

//This function picks strongest unmarked peak and cheks fitness to a comb functions corresponding to charges given with Deconvolution Params.
//The most fitting peaks are put into a cluster.
func (pks Peaks) Deconvolution(params DeconvolutionParams) (clusts Clusters) {
	////////////////////////Consts and helpers start from here////////////////////////
	const ThreshIntensVsMainPeak = 0.1

	findMax := func(slice []float64) (idxMax int) {
		idxMax = 0
		for i := range slice {
			if slice[i] > slice[idxMax] {
				idxMax = i
			}
		}
		return
	}
	////////////////////////Consts and helpers end here////////////////////////

	trimed := pks.noizeCut(params.IgnoreLevel)
	trimed.SortByMPerZ()

	idConv := trimed.IDConversion()

	//multiCharge corresponds to signals corrected by charge and cation mass
	//multicharge[i][j] corresponds to j-th peak of (ChargeMin+i)-charge spectrum
	multiCharge := make([]Peaks, params.ChargeMax-params.ChargeMin+1)
	for i := range multiCharge {
		multiCharge[i] = make(Peaks, trimed.Length())
		for j := range multiCharge[i] {
			multiCharge[i][j] = trimed[j]
			multiCharge[i][j].MPerZ = (trimed[j].MPerZ - params.AdditionalMass) * float64(params.ChargeMin+i)
		}
	}

	idxMiddle := (params.ClusterPeaksMin - 1) / 2
	numBackward := idxMiddle
	numForward := (params.ClusterPeaksMin - 1) - idxMiddle

	trimed.SortByIntensReverse()
	unchecked := 0

	for numClust := 0; numClust <= params.ClustersMax-1; /*numClust++*/ /*TODO: numClust is increased only if the roop suceeded to make a cluseter*/ {
		//TODO: check the fitting of the unchecked peak with an ID of trimed[unchecked].ID through all charges.

	}
	return
}

//Clusterizer picks peaks to form a cluster by expanding a designated peak. The peaks pks has to be sorted in tems of m/z
func clusterizer(pksSlice []Peaks, minCharge int, target Peak, backward, forward int, params DeconvolutionParams) (retCluster Cluster) {

	///////////////////////// Working variables start from here/////////////////////////
	accumulation := make([]float64, len(pksSlice))
	clusterPksIDs := make([]order.SortableInt, len(pksSlice))
	///////////////////////// Working variables ends here/////////////////////////

	for idxPksSlic := range pksSlice {
		accumulation[idxPksSlic], clusterPksIDs[idxPksSlic] = pksSlice[idxPksSlic].fitByID(target.ID, backward, forward, params)
	}

	var idxAccumBiggest int
	var valAccumBiggest float64

	idxAccumBiggest = ClusterNotFormed
	for idxPksSlic := range pksSlice {
		if accumulation[idxPksSlic] > valAccumBiggest {
			idxAccumBiggest = idxPksSlic
			valAccumBiggest = accumulation[idxPksSlic]
		}
	}

	if idxAccumBiggest == ClusterNotFormed {
		retCluster.peaks = nil
		retCluster.dominantCharge = ClusterNotFormed
		retCluster.monoisotopic = ClusterNotFormed
		retCluster.mostAbundant = ClusterNotFormed
		retCluster.obsCharge = nil
	} else {
		retCluster.mostAbundant, clusterPksIDs[idxAccumBiggest], _, _ = pksSlice[idxAccumBiggest].expand(target.MPerZ, target.ID, clusterPksIDs[idxAccumBiggest], backward, forward, params)
		retCluster.dominantCharge = minCharge + idxAccumBiggest
		idxMonoisotopic := (clusterPksIDs[idxAccumBiggest])[0]
		retCluster.monoisotopic = (pksSlice[idxAccumBiggest])[idxMonoisotopic].MPerZ
		for _, idxPeak := range clusterPksIDs[idxAccumBiggest] {
			retCluster.peaks = append(retCluster.peaks, (pksSlice[idxAccumBiggest])[idxPeak])
		}
	}

	return

}

func (pks Peaks) noizeCut(noizeLevel float64) (trimmed Peaks) {
	for i := range pks {
		if pks[i].Intens >= noizeLevel {
			trimmed = append(trimmed, pks[i])
		}
	}
	return
}

//fitByID calculates fitting with a comb.
//Sort the pks before use in m/z order
func (pks Peaks) fitByID(targetID int, backward int, forward int, params DeconvolutionParams) (accumulation float64, clusterPksID order.SortableInt) {
	additionalMass := params.AdditionalMass
	maxError := params.MPerZErrMax

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

//fitbyMperZ calculates fitting with a comb.
//Sort the pks before use in m/z order. If peak picking fails, returned cluster PksID == nil
func (pks Peaks) fitByMPerZ(targetMPerZ float64, backward int, forward int, params DeconvolutionParams) (accumulation float64, clusterPksID order.SortableInt) {
	additionalMass := params.AdditionalMass
	maxError := params.MPerZErrMax

	//TODO: picked peak has to be smaller than the target MPerZ
	//TODO: an Error has to be returned if peak picking fails

	//	idSeq := pks.IDConversion()
	//	targetIndex := idSeq[targetID]

	clusterPksID = make(order.SortableInt, 1)
	match := pks.BinarySearchMPerZ(targetMPerZ, maxError)
	match.SortByIntensReverse()
	clusterPksID[0] = match[0].Cluster

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

func (pks Peaks) optim(targetMPerZ float64, targetID int, clusterPksID order.SortableInt, additionalMass float64) (optimTargetMPerZ float64, stdev float64) {

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

//expand MPerZ expands a cluster. clusterPksID has to be sorted ID-wise prior to use.
func (pks Peaks) expand(targetMPerZ float64, targetID int, clusterPksID order.SortableInt, backward int, forward int,
	decParam DeconvolutionParams) (retTargetMPerZ float64, retClusterPksID order.SortableInt, retBackward int, retForward int) {

	const errPctRange = 90

	pks.SortByID()
	//	idSeq := pks.IDConversion()

	order.QuickSort(clusterPksID)

	deltaMPerZ := decParam.AdditionalMass

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
				targetMPerZ, stdev = pks.optim(targetMPerZ, targetID, clusterPksID, decParam.AdditionalMass)
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
				targetMPerZ, stdev = pks.optim(targetMPerZ, targetID, clusterPksID, decParam.AdditionalMass)
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
