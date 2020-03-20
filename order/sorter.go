package order

//Sorter is a interface whose element's magnitude can be evaluated and be swapped with another element
type Sorter interface {
	Compare(int, int) int
	Swap(int, int)
	Length() int
}
