package mass

import (
	"PainTheMaster/mybraly/order"
)

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////        Peaks       //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

//Peak represents each peaks in the mass spectrum
type Peak struct {
	ID      int // ID has to start from 0 and continuous. The sequence of ID has to be identical to MPerZ
	MPerZ   float64
	Intens  float64
	Cluster int
	OverLap bool
}

const (
	//ClusterUnAssigned is a claster for unassigned peak
	ClusterUnAssigned = -1
)

//Peaks is a slice of Peaks
type Peaks []Peak

//Reset resets peaks. This function assigns an ID to each peak, reset Cluster to -1 (belonging to no cluster) and Overlap to false
//Before resetting only MPerZ and Intens is needed
func (pks Peaks) Reset() {
	pks.SortByMPerZ()
	for i := range pks {
		pks[i].ID = i
		pks[i].Cluster = ClusterUnAssigned
		pks[i].OverLap = false
	}
}

//Compare of the Peaks compare element i and j. If the element i is bigger thant j, this returns 1
// if equal 0, and if smaller -1
func (pks Peaks) Compare(i int, j int) (result int) {
	if pks[i].MPerZ > pks[j].MPerZ {
		result = 1
	} else if pks[i].MPerZ < pks[j].MPerZ {
		result = -1
	} else {
		result = 0
	}
	return
}

// Swap swaps the element i and j
func (pks Peaks) Swap(i int, j int) {
	pks[i], pks[j] = pks[j], pks[i]
}

//Length returns the number of the elements of the pks
func (pks Peaks) Length() (length int) {
	length = len(pks)
	return
}

//SortByIntens sorts the pks by intensity
func (pks Peaks) SortByIntens() {

	compareInt := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		if asserted[i].Intens > asserted[j].Intens {
			result = 1
		} else if asserted[i].Intens < asserted[j].Intens {
			result = -1
		} else {
			result = 0
		}
		return
	}

	order.PartialQuickSortFunc(pks, 0, pks.Length()-1, compareInt)
}

//SortByIntensReverse sorts the pks by intensity
func (pks Peaks) SortByIntensReverse() {

	compareIntReverse := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		if asserted[i].Intens > asserted[j].Intens {
			result = -1
		} else if asserted[i].Intens < asserted[j].Intens {
			result = 1
		} else {
			result = 0
		}
		return
	}

	order.PartialQuickSortFunc(pks, 0, pks.Length()-1, compareIntReverse)
}

//SortByMPerZ sorts the pks by m/z
func (pks Peaks) SortByMPerZ() {
	compareMPerZ := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		if asserted[i].MPerZ > asserted[j].MPerZ {
			result = 1
		} else if asserted[i].MPerZ < asserted[j].MPerZ {
			result = -1
		} else {
			result = 0
		}
		return
	}
	order.PartialQuickSortFunc(pks, 0, pks.Length()-1, compareMPerZ)
}

//SortByID sorts the pks by ID
func (pks Peaks) SortByID() {
	compareID := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		if asserted[i].ID > asserted[j].ID {
			result = 1
		} else if asserted[i].ID < asserted[j].ID {
			result = -1
		} else {
			result = 0
		}
		return
	}
	order.PartialQuickSortFunc(pks, 0, pks.Length()-1, compareID)
}

//SortByCluster sorts pks by cluster unassigned pekas are corrected in a ragne from 0 to "lastIdxUnassigned"
func (pks Peaks) SortByCluster() {
	compareCluster := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		if asserted[i].ID > asserted[j].ID {
			result = 1
		} else if asserted[i].ID < asserted[j].ID {
			result = -1
		} else {
			if asserted[i].ID == ClusterUnAssigned {
				if asserted[i].Intens > asserted[j].Intens {
					result = -1
				} else if asserted[i].Intens < asserted[j].Intens {
					result = 1
				} else {
					result = 1
				}
			} else {
				if asserted[i].MPerZ > asserted[j].MPerZ {
					result = 1
				} else if asserted[i].MPerZ < asserted[j].MPerZ {
					result = -1
				} else {
					result = 0
				}
			}
		}
		return
	}

	order.PartialQuickSortFunc(pks, 0, pks.Length()-1, compareCluster)
}

//SortByMPerZProxim sorts pks according to proximity to target m/z
func (pks Peaks) SortByMPerZProxim(targetMPerZ float64) {
	compareProximity := func(srt order.Sorter, i int, j int) (result int) {
		asserted := srt.(Peaks)
		proxISq := asserted[i].MPerZ - targetMPerZ
		proxISq *= proxISq

		proxJSq := asserted[j].MPerZ - targetMPerZ
		proxJSq *= proxISq

		if proxISq > proxJSq {
			result = 1
		} else if proxISq < proxJSq {
			result = -1
		} else {
			if asserted[i].Intens > asserted[j].Intens {
				result = -1
			} else if asserted[i].Intens < asserted[j].Intens {
				result = 1
			} else {
				result = 0
			}
		}
		return
	}
	order.PartialQuickSortFunc(pks, 0, pks.Length(), compareProximity)
}

//IDConversion returns a int slice in which ID-index table is contained.
//idSeq[ID] == index in pks
func (pks Peaks) IDConversion() (idSeq []int) {
	idSeq = make([]int, pks.Length())
	for idxPks := range pks {
		idSeq[pks[idxPks].ID] = idxPks
	}
	return
}

//BinarySearchMPerZ of pks finds peaks that matches target within the maxerr range
//The peaks has to be sorted before being subjected to this search function
func (pks Peaks) BinarySearchMPerZ(target float64, maxerr float64) (match Peaks) {
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
	middle := left + (right-left)/2

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
			middle = left + (right-left)/2
		} else if compareWithError(middle) < 0 {
			right = middle
			middle = left + (right-left)/2
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
	match.SortByMPerZProxim(target)
	return
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////       Cluster      //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Clusters represents clustrs of peaks correpsond to one chemical speceis in multiple valences
type Clusters struct {
	peaks          Peaks
	monoisotopic   float64
	mostAbundant   float64
	obsCharge      []int
	dominantCharge int
}
