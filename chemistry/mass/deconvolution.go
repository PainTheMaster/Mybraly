package mass

//DeconvolutionParams is a bundle of parameters for deconvolution
type DeconvolutionParams struct {
	MinimumClusters int
	ChargeMin       int
	ChargeMax       int
	MinimumPeaks    int
	additionalMass  float64
	MaxMPerZErr     float64
	NoizeLevel      float64
}

//Clusterizer clusters peaks from pks
func Clusterizer(pks Peaks, nimimumPeaks int, maxError float64, noizeLevel float64, chargeMin int, chargeMax int, additionalMass float64) {
	pks.SortByMPerZ()

	//multiCharge corresponds to signals corrected by charge and cation mass
	multiCharge := make([]Peaks, chargeMax-chargeMin+1)
	for i := range multiCharge {
		multiCharge[i] = make(Peaks, pks.Length())
		for j := range multiCharge[i] {
			multiCharge[i][j] = pks[j]
			multiCharge[i][j].MPerZ = (pks[j].MPerZ - additionalMass) * float64(chargeMin+i)
		}
	}

	pks.SortByIntensReverse()
}

func binarySearchMPerZ(pks Peaks, target float64, maxerr float64) (match Peaks) {
	const boundaryToLinear = 4

	//cmpreWithError returns 1 if target is biggerthan the element i
	//returns -1 if target is smaller, and 0 if equal
	compareWithError := func(i int) (result int) {
		if target > pks[i].MPerZ+maxerr {
			result = 1
		} else if target < pks[i].MPerZ-maxerr {
			result = -1
		} else {
			result = 0
		}
		return
	}

	linearSearch := func(searchFrom int, searchTo int) (result int) {
		const resultNotFound = -1
		for idxSearch := searchFrom; idxSearch <= searchTo; idxSearch++ {
			if compareWithError(idxSearch) == 0 {
				result = idxSearch
				break
			} else if idxSearch == searchTo && compareWithError(idxSearch) != 0 {
				result = resultNotFound
			}
		}
		return
	}

	left := 0
	right := pks.Length() - 1
	middle := (left + right) / 2

	match = nil

	from := -1
	to := -1

	for {
		if to-from+1 <= boundaryToLinear {
			idxMatch := linearSearch(from, to)
			if idxMatch < 0 {
				from, to = -1, -1
			} else {
				from = idxMatch
				to = idxMatch

				for compareWithError(from-1) == 0 {
					from--
				}
				for compareWithError(to+1) == 0 {
					to++
				}
				break
			}
		} else if compareWithError(middle) > 0 {
			left = middle
			middle = (left + right) / 2
		} else if compareWithError(middle) < 0 {
			right = middle
			middle = (left + right) / 2
		} else {
			from = middle
			to = middle

			for compareWithError(from-1) == 0 {
				from--
			}
			for compareWithError(to+1) == 0 {
				to++
			}
			break
		}
	}

	if from >= 0 {
		match = make(Peaks, to-from+1)
		for i := 0; from+i <= to; i++ {
			match[i] = pks[from+i]
		}
	}
	return
}