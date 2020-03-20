package massspec

import "PainTheMaster/mybraly/chemistry"

//IsotopePatternCombination combines two isotope pattern
func IsotopePatternCombination(isotopePatternTo, isotopePatternFrom chemistry.Isotopes) (isotopePattern chemistry.Isotopes) {
	isotopePattern = isotopePatternTo

	tempPattern := make([]chemistry.Isotopes, isotopePatternFrom.Length())
	for row := range tempPattern {
		tempPattern[row] = make(chemistry.Isotopes, isotopePattern.Length())
		for idxPeaks := range isotopePattern {
			tempPattern[row][idxPeaks] = isotopePattern[idxPeaks]
		}
	}
	//Now, combine: each isotopes
	for idxAddIsotopeFrom, addIsotopeFrom := range isotopePatternFrom {
		//Manipulating all preexisting isotopologues: adding the mass and calculating the abundance
		for idxAddIsotopeTo := range tempPattern[idxAddIsotopeFrom] {
			tempPattern[idxAddIsotopeFrom][idxAddIsotopeTo].Abundance *= addIsotopeFrom.Abundance
			tempPattern[idxAddIsotopeFrom][idxAddIsotopeTo].Mass += addIsotopeFrom.Mass
		}
	}
	//Here, temporal pattern has to be marged, and isotopePattern is update
	for idxMerged := 1; idxMerged <= len(tempPattern)-1; idxMerged++ {
		isotopeMarge(&tempPattern[0], tempPattern[idxMerged])
	}
	isotopePattern = tempPattern[0]
	return
}

//CompositionIsotopePattern calculates isotope pattern of a collection of atoms with a ChemComponent C
func CompositionIsotopePattern(C chemistry.ChemComposition) (isotopePattern chemistry.Isotopes) {
	//	C[0].Elem
	isotopePattern = make(chemistry.Isotopes, 1)
	isotopePattern[0].Mass = 0.0
	isotopePattern[0].Abundance = 1.0

	//Each composing element: count and sort
	for _, component := range C {
		componentCount := component.Count
		componentIsotopePatt := component.Elem.IsotopPattern
		//Each atom of the element
		for coutAtom := 1; coutAtom <= componentCount; coutAtom++ {
			//Each newly added isotope
			isotopePattern = IsotopePatternCombination(isotopePattern, componentIsotopePatt)
		}
	}
	return isotopePattern
}
