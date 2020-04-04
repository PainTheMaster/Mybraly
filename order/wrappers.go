package order

//SortableIntSlice is a slice of int that is sortable
type SortableIntSlice []int

//Compare compares
func (sis SortableIntSlice) Compare(i int, j int) (result int) {
	if sis[i] > sis[j] {
		result = 1
	} else if sis[i] < sis[j] {
		result = -1
	} else {
		result = 0
	}
	return
}

//Swap swaps
func (sis SortableIntSlice) Swap(i int, j int) {
	sis[i], sis[j] = sis[j], sis[i]
}

//Length returns length of the slice
func (sis SortableIntSlice) Length() int {
	return len(sis)
}
