package chemistry

import "strings"

//ChemCollectionInterf is not a single existance like "an atom" or "a molecule", but element as a bundle of atoms or molecules in a flask
type ChemCollectionInterf interface {
	Symbol() string
	IsotopePattern() Isotopes
	FormularWeight() float64
}

//ChemCollection is a bundle of chemical characteristics common to atoms, fragments and so on
type ChemCollection struct {
	Symbol         string
	IsotopePattern Isotopes
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
	if isotopes[i].Mass > isotopes[j].Mass {
		return 1
	} else if isotopes[i].Mass < isotopes[j].Mass {
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

//CompareAbund compares the abundance of the constituting peaks of isotopes
//If the member i is biggger j, it retuurns 1, if equal 0, and if smaller -1
func (isotopes Isotopes) CompareAbund(i int, j int) int {
	var ans int
	if isotopes[i].Abundance > isotopes[j].Abundance {
		ans = 1
	} else if isotopes[i].Abundance < isotopes[j].Abundance {
		ans = -1
	} else {
		ans = 0
	}
	return ans
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

//Compare of ChemComposition compares the order of the componetn element i and j.
//An element with higher priority has smaller number.
//if component i is H and j is N, Compare(j, i) = 1
func (C ChemComposition) Compare(i int, j int) (ans int) {
	const (
		orderC = iota
		orderH
		orderO
		orderN
		orderOther
	)

	elemTyping := func(symbol string) (order int) {
		switch symbol {
		case "C":
			order = orderC
		case "H":
			order = orderH
		case "O":
			order = orderO
		case "N":
			order = orderN
		default:
			order = orderOther
		}
		return
	}

	if i == j {
		ans = 0
	} else {
		symbolI := C[i].Elem.Symbol
		symbolJ := C[j].Elem.Symbol
		typeI := elemTyping(symbolI)
		typeJ := elemTyping(symbolJ)

		if typeI < orderOther {
			if typeI-typeJ > 0 {
				ans = 1
			} else if typeI-typeJ < 0 {
				ans = -1
			} else {
				ans = 0
			}
		} else if typeJ < orderOther {
			ans = 1
		} else {
			ans = strings.Compare(symbolI, symbolJ)
		}
	}
	return
}

// Swap swaps the component i and j
func (C ChemComposition) Swap(i int, j int) {
	C[i], C[j] = C[j], C[i]
}

//Length returns the number of componets
func (C ChemComposition) Length() int {
	return len(C)
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////        Structures      ////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Structures is a inerface that is common to entities with structures
//Structures has to haave a ChemComposition() that returns the composition of the structure
type Structures interface {
	ChemComposition() ChemComposition
	IsotopePattern() Isotopes
	FormularWeight() float64
}

//Fragment represents a fragment
type Fragment struct {
	ChemCollection
	Composition ChemComposition
}

//ChemComposition of a fragment returns ChemComposition of the fragment
func (frag Fragment) ChemComposition() ChemComposition {
	return frag.Composition
}

//IsotopePattern returns isotope pattern of the fragment
func (frag Fragment) IsotopePattern() Isotopes {
	return frag.ChemCollection.IsotopePattern
}

//FormularWeight returns formular weight of the fragment
func (frag Fragment) FormularWeight() float64 {
	return frag.ChemCollection.FormularWeight
}

//Fragments is a collection of Fragments
type Fragments []Fragment

//Molecule represents a molecule
type Molecule struct {
	ChemCollection
	Fragments   Fragments
	Composition ChemComposition
}

//ChemComposition of a molecule returns ChemComposition of the fragment
func (molec Molecule) ChemComposition() ChemComposition {
	return molec.Composition
}

//IsotopePattern returns isotope pattern of the molecule
func (molec Molecule) IsotopePattern() Isotopes {
	return molec.ChemCollection.IsotopePattern
}

//FormularWeight returns formular weight of the molecule
func (molec Molecule) FormularWeight() float64 {
	return molec.ChemCollection.FormularWeight
}
