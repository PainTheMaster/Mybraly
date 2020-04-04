package mass

import (
	"PainTheMaster/mybraly/order"
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
func (pks Peaks) fit(targetID int, interval int, additionalMass float64, backward int, forward int, maxError float64) (accumulation float64, clusterPksID order.SortableIntSlice) {
	idSeq := pks.IDConversion()
	targetIndex := idSeq[targetID]

	clusterPksID = make(order.SortableIntSlice, 1)
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

func (pks Peaks) optim(targetID int, clusterPksID order.SortableIntSlice, interval int, additionalMass float64) (optimTargetMPerZ float64) {

	idSeq := pks.IDConversion()

	order.QuickSort(clusterPksID)

	customCompare := func(srt order.Sorter, toBeComp int) (result int) {
		asserted := srt.(order.SortableIntSlice)
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

	makeComb := func(targetMPerZ float64, backward int, forward int, additionalMass float64) (comb []float64) {
		comb = make([]float64, backward+forward+1)
		for i := 0; i <= backward+forward; i++ {
			comb[i] = targetMPerZ + additionalMass*float64(i-backward)
		}
		return
	}

	fitness := func(comb []float64) (accumError float64) {
		accumError = 0.0
		for i := 0; i <= clusterPksID.Length()-1; i++ {
			accumError += math.Abs(comb[i] - pks[idSeq[targetID]].MPerZ)
		}
		return
	}

	const (
		shiftRatio   = 0.001
		correctRatio = 0.1
		repetition   = 10
	)

	targetMPerZ := pks[idSeq[targetID]].MPerZ

	backward := targetIdx
	forward := (clusterPksID.Length() - 1) - targetIdx

	for cycle := 1; cycle <= repetition; cycle++ {
		comb := makeComb(targetMPerZ, backward, forward, additionalMass)
		shiftedMPerZ := targetMPerZ * (1.0 + shiftRatio)
		shiftedComb := makeComb(shiftedMPerZ, backward, forward, additionalMass)

		accumError := fitness(comb)
		shiftedAccumError := fitness(shiftedComb)

		differential := (shiftedAccumError - accumError) / (targetMPerZ * shiftRatio)
		targetMPerZ -= correctRatio * differential
	}

	optimTargetMPerZ = targetMPerZ
	return
}
