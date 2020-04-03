package mass

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
func (pks Peaks) fit(targetID int, interval int, additionalMass float64, backward int, forward int, maxError float64) (accumulation float64, tempCluster Peaks) {
	idSeq := pks.IDConversion()
	targetIndex := idSeq[targetID]

	tempCluster = make(Peaks, 1)
	tempCluster[0] = pks[targetIndex]

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
				tempCluster = append(tempCluster, match[j])
				break
			}
		}
	}

	return
}
