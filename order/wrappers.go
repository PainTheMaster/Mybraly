package order

//SortableInt is a slice of int that is sortable
type SortableInt []int

//Compare compares
func (sis SortableInt) Compare(i int, j int) (result int) {
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
func (sis SortableInt) Swap(i int, j int) {
	sis[i], sis[j] = sis[j], sis[i]
}

//Length returns length of the slice
func (sis SortableInt) Length() int {
	return len(sis)
}
