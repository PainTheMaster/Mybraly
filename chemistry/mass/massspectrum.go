package mass

import (
	"PainTheMaster/mybraly/order"
)

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////        Peaks       //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

//Peak represents each peaks in the mass spectrum
type Peak struct {
	ID        int // ID has to start from 0 and continuous. The sequence of ID has to be identical to MPerZ
	MPerZ     float64
	Intens    float64
	ClusterID int
	Overlap   bool
}

const (
	//ClusterUnAssigned is a claster for unassigned peak
	ClusterUnAssigned = -1
	//SearchNotFound is used if a result is not found in a search
	SearchNotFound = -1
)

//Peaks is a slice of Peaks
type Peaks []Peak

//Reset resets peaks. This function assigns an ID to each peak, reset Cluster to -1 (belonging to no cluster) and Overlap to false
//Before resetting only MPerZ and Intens is needed
func (pks Peaks) Reset() {
	pks.SortByMPerZ()
	for i := range pks {
		pks[i].ID = i
		pks[i].ClusterID = ClusterUnAssigned
		pks[i].Overlap = false
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
	const boundaryToLinear = 16

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

		for idxSearch := searchFrom; idxSearch <= searchTo; idxSearch++ {
			if compareWithError(idxSearch) == 0 {
				result = idxSearch
				break
			} else if idxSearch == searchTo && compareWithError(idxSearch) != 0 {
				result = SearchNotFound
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
		if right-left+1 <= boundaryToLinear {
			idxMatch := linearSearch(left, right)
			if idxMatch == SearchNotFound {
				from, to = SearchNotFound, SearchNotFound
			} else {
				from = idxMatch
				to = idxMatch

				for compareWithError(from-1) == 0 && (from-1) >= 0 {
					from--
				}
				for compareWithError(to+1) == 0 && (to+1) <= pks.Length()-1 {
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

	if from != SearchNotFound {
		match = make(Peaks, to-from+1)
		for i := 0; from+i <= to; i++ {
			match[i] = pks[from+i]
		}
	}
	match.SortByMPerZProxim(target)
	return
}

//BinarySearchID find and returns an index of a peak whose ID matches an argument id
//pks has to be sorted ID-wise beforehand
func (pks Peaks) BinarySearchID(id int) (idxResult int) {
	const boundaryToLinear = 16

	idxResult = SearchNotFound

	linearSearch := func(searchFrom, searchTo int) (idxResult int) {
		for idxSearch := searchFrom; idxSearch <= searchTo; idxSearch++ {
			if pks[idxSearch].ID == id {
				idxResult = idxSearch
				break
			} else if idxSearch == searchTo && pks[idxSearch].ID != id {
				idxResult = SearchNotFound
			}
		}
		return
	}

	left := 0
	right := pks.Length()
	middle := left + (right-left)/2

	for {
		if right-left+1 <= boundaryToLinear {
			idxResult = linearSearch(left, right)
			break
		} else if id < pks[middle].ID {
			right = middle
			middle = left + (right-left)/2
		} else if pks[middle].ID < id {
			left = middle
			middle = left + (right-left)/2
		} else {
			idxResult = middle
			break
		}
	}
	return
}

//DeleteIdx deletes an element (*pks)[idx] and make ptrPks shorter.
func (pks *Peaks) DeleteIdx(idx int) {
	len := (*pks).Length()

	left := (*pks)[0:idx:idx]
	right := (*pks)[idx+1 : len : (len - 1 - idx)]
	*pks = append(left, right...)
}

//DeleteID deletes an peak whose ID matches an arugment id
func (pks *Peaks) DeleteID(id int) {
	idxDel := (*pks).BinarySearchID(id)
	pks.DeleteIdx(idxDel)
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////       Cluster      //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Cluster represents clustrs of peaks correpsond to one chemical speceis in multiple valences
type Cluster struct {
	peaks          Peaks
	monoisotopic   float64
	mostAbundant   float64
	obsCharge      []int
	dominantCharge int
}

//Clusters is a slice of cluster
type Clusters []Cluster

//Compare compares i and j
func (clusts Clusters) Compare(i int, j int) (result int) {
	if clusts[i].monoisotopic > clusts[j].monoisotopic {
		result = 1
	} else if clusts[i].monoisotopic < clusts[j].monoisotopic {
		result = -1
	} else {
		result = 0
	}
	return
}

//Swap swaps
func (clusts Clusters) Swap(i int, j int) {
	clusts[i], clusts[j] = clusts[j], clusts[i]
}

//Length returns the length of the clusters
func (clusts Clusters) Length() int {
	return len(clusts)
}
