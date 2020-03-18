package order

//Sorter is a interface whose element's magnitude can be evaluated and be swapped with another element
type Sorter interface {
	DefaultMagnitude(int) float64
	Swap(int, int)
	Length() int
}
