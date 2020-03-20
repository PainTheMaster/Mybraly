package massspec

import "PainTheMaster/mybraly/chemistry"

//CompositionIsotopePatternPlain calculates isotope pattern of a collection of atoms with a ChemComponent C
func CompositionIsotopePatternPlain(C chemistry.ChemComposition) (isotopePattern chemistry.Isotopes) {
	//	C[0].Elem
	isotopePattern = make(chemistry.Isotopes, 1)
	isotopePattern[0].Mass = 0.0
	isotopePattern[0].Abundance = 1.0

	//Each composing element: count and sort
	for _, component := range C {
		countComp := component.Count
		compIsotopePatt := component.Elem.IsotopPattern
		//Each atom of the element
		for coutAtom := 1; coutAtom <= countComp; coutAtom++ {
			//Each newly added isotope
			tempPattern := make([]chemistry.Isotopes, compIsotopePatt.Length())
			for row := range tempPattern {
				tempPattern[row] = make(chemistry.Isotopes, len(isotopePattern))
				for idxPeaks := range isotopePattern {
					tempPattern[row][idxPeaks] = isotopePattern[idxPeaks]
				}
			}
			//Now, add atoms: each isotopes
			for idxAddIsotope, addIsotope := range compIsotopePatt {
				//Manipulating all preexisting isotopologues: adding the mass and calculating the abundance
				for idxPeaks := range tempPattern[idxAddIsotope] {
					tempPattern[idxAddIsotope][idxPeaks].Abundance *= addIsotope.Abundance
					tempPattern[idxAddIsotope][idxPeaks].Mass += addIsotope.Mass
				}
			}
			//Here, temporal pattern has to be marged, and isotopePattern is update
			for idxMerged := 1; idxMerged <= len(tempPattern)-1; idxMerged++ {
				DummyIsotopeMarge(&tempPattern[0], tempPattern[idxMerged])
			}
			isotopePattern = tempPattern[0]
		}
	}
	return isotopePattern
}
