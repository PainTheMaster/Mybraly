package chemistry

import (
	"math"
	"strings"
)

const intSize = 32 << (^uint(0) >> 63)

//Reaction combines mol1 and mol3 and returns mol3
func Reaction(mol1, mol2 Structures) (mol3 Molecule) {

	maxPeakNumCalc := func() int {
		const intMax = int(^uint(0) >> 1)
		maxPeakNum := int(math.Sqrt(float64(intMax)))
		return maxPeakNum
	}

	maxIsotpeNum := maxPeakNumCalc()
	var isotopes1, isotopes2 Isotopes
	isotopes2, isotopes2 = mol1.IsotopePattern(), mol2.IsotopePattern()

	if isotopes1.Length() > maxIsotpeNum {
		IsotopePatternCutOffNum(&isotopes1, maxIsotpeNum)
	}
	if isotopes2.Length() > maxIsotpeNum {
		IsotopePatternCutOffNum(&isotopes2, maxIsotpeNum)
	}

	mol3.ChemCollection.IsotopePattern = IsotopePatternCombination(isotopes1, isotopes2)
	IsotopePatternNormalization(&(mol3.ChemCollection.IsotopePattern))

	mol3.Composition = combineChemComposition(mol1.ChemComposition(), mol2.ChemComposition())

	mol3.ChemCollection.FormularWeight = mol1.FormularWeight() + mol2.FormularWeight()

	return
}

func combineChemComposition(cc1, cc2 ChemComposition) (combined ChemComposition) {

	combined = make(ChemComposition, cc1.Length())
	for idxCopy, componentCopiedFrom := range cc1 {
		combined[idxCopy].Elem = componentCopiedFrom.Elem
		combined[idxCopy].Count = componentCopiedFrom.Count
	}

	for _, ccAddedFrom := range cc2 {
		boolFound, idxFound := componentMatch(combined, ccAddedFrom)
		if boolFound {
			combined[idxFound].Count += ccAddedFrom.Count
		} else {
			combined = append(combined, ccAddedFrom)
		}
	}
	return
}

func componentMatch(compos ChemComposition, component ChemComponent) (boolFound bool, idxFound int) {
	var componentPool ChemComponent
	boolFound = false
	for idxFound, componentPool = range compos {
		if strings.Compare(componentPool.Elem.Symbol, component.Elem.Symbol) == 0 {
			boolFound = true
			break
		}
	}
	if !boolFound {
		idxFound = -1
	}
	return
}
