package order

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////SortableInt from here///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//SortableInt is a slice of int that is sortable
type SortableInt []int

//Compare compares
func (si SortableInt) Compare(i int, j int) (result int) {
	if si[i] > si[j] {
		result = 1
	} else if si[i] < si[j] {
		result = -1
	} else {
		result = 0
	}
	return
}

//Swap swaps
func (si SortableInt) Swap(i int, j int) {
	si[i], si[j] = si[j], si[i]
}

//Length returns length of the slice
func (si SortableInt) Length() int {
	return len(si)
}

//Sort sorts the SortableInt si
func (si SortableInt) Sort() {
	QuickSort(si)
}

//SortReverse sorts the Sortable int si reversely
func (si SortableInt) SortReverse() {
	QuickSortReverse(si)
}

///////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////SortableInt to here////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////SortableFloat64 from here/////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

//SortableFloat64 is is a sortable []float64
type SortableFloat64 []float64

//Compare compares
func (sf SortableFloat64) Compare(i int, j int) (result int) {
	if sf[i] > sf[j] {
		result = 1
	} else if sf[i] < sf[j] {
		result = -1
	} else {
		result = 0
	}
	return
}

//Swap swaps
func (sf SortableFloat64) Swap(i int, j int) {
	sf[i], sf[j] = sf[j], sf[i]
}

//Length returns length of the slice
func (sf SortableFloat64) Length() int {
	return len(sf)
}

//Sort sorts the SortablFloat64 sf
func (sf SortableFloat64) Sort() {
	QuickSort(sf)
}

//SortReverse sorts the Sortable int sf reversely
func (sf SortableFloat64) SortReverse() {
	QuickSortReverse(sf)
}

///////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////SortableFloat64 to here//////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////