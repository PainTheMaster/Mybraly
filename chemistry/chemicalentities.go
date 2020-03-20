package chemistry

//ChemCollectionInterf is not a single existance like "an atom" or "a molecule", but element as a bundle of atoms or molecules in a flask
type ChemCollectionInterf interface {
	Symbol() string
	IsotopePattern() Isotopes
	FormularWeight() float64
}

//ChemCollection is a bundle of chemical characteristics common to atoms, fragments and so on
type ChemCollection struct {
	Symbol         string
	IsotopPattern  Isotopes
	FormularWeight float64
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////       Isotope      ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Isotope is a composite of Mass and Abundance
type Isotope struct {
	MassNumber int
	Mass       float64
	Abundance  float64
}

//Isotopes is a slice of Isotopes. This is a Sorter
type Isotopes []Isotope

//Compare of Isotopes type returns 1 if the Mass number of member i is bigger than that of member j, 0 if the same (which is not likely to happen),and -1 if smaller.
func (isotopes Isotopes) Compare(i int, j int) int {
	if isotopes[i].MassNumber > isotopes[j].MassNumber {
		return 1
	} else if isotopes[i].MassNumber < isotopes[j].MassNumber {
		return -1
	} else {
		return 0
	}
}

//Swap of the Isotopes type swaps the member i and j
func (isotopes Isotopes) Swap(i int, j int) {
	isotopes[i], isotopes[j] = isotopes[j], isotopes[i]
}

//Length of the Isotopes type returns the number of the isotopes
func (isotopes Isotopes) Length() int {
	return len(isotopes)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////       Element      ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Element is a element
type Element struct {
	AtomNumber int
	ChemCollection
}

//PeriodicTable is a collection of Elements
//PeriodicTable is sortable
type PeriodicTable []Element

//Compare of PeriodicTable comapres the element by atomic number
//It returns 1 if atomic number of the i-th element is bigger than j,
//0 if the same, and -1 if smaller
func (pt PeriodicTable) Compare(i int, j int) int {
	if pt[i].AtomNumber > pt[j].AtomNumber {
		return 1
	} else if pt[i].AtomNumber < pt[j].AtomNumber {
		return -1
	} else {
		return 0
	}
}

//Swap of PeriodicTable swaps
func (pt PeriodicTable) Swap(i int, j int) {
	pt[i], pt[j] = pt[j], pt[i]
}

//Length of PeriodicTable returns the number of the members
func (pt PeriodicTable) Length() int {
	return len(pt)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////Foundation of atoms ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//ChemComponent is a set of element and number of it in a molecule
type ChemComponent struct {
	Elem  Element
	Count int
}

//ChemComposition is s slice of ChemComponents. ChemComposition is a Sorter
type ChemComposition []ChemComponent

/*
func (C ChemComposition) DefaultMagnitude(i int) float64 {

}
*/

/*
Swap(int, int)
Length() int
*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////        Atoms       ////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Molecule represents a molecule
type Molecule struct {
	ChemCollection
}
