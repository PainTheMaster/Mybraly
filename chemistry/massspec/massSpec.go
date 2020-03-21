package chemistry

import (
	"PainTheMaster/mybraly/chemistry"
	"PainTheMaster/mybraly/order"
)

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

//IosotpePatternCutOffNum cuts off the IsotopePattern cuts off *prtIsotPattern by intensity.
//Only top num peaks survive.
func IosotpePatternCutOffNum(prtIsotPattern *chemistry.Isotopes, num int) {
	isoPattern := *prtIsotPattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(chemistry.Isotopes)
		ans = asserted.CompareAbund(i, j) * (-1)
		return
	}

	order.PartialQuickSortFunc(isoPattern, 0, isoPattern.Length()-1, surrogateFuncAbundReverse)

	isoPattern = isoPattern[0:num:num]

	order.QuickSort(isoPattern)

	*prtIsotPattern = isoPattern
}

//IosotpePatternCutOffTotalAbund cuts off the by total abundance percentage.
//Maximum sum of discarded peaks is not more than discardedAbund
func IosotpePatternCutOffTotalAbund(prtIsotPattern *chemistry.Isotopes, pctDiscardedAbund float64) {
	isoPattern := *prtIsotPattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(chemistry.Isotopes)
		ans = asserted.CompareAbund(i, j) * (-1)
		return
	}

	order.PartialQuickSortFunc(isoPattern, 0, isoPattern.Length()-1, surrogateFuncAbundReverse)

	var totalAbund float64
	totalAbund = 0.0
	for _, isotope := range isoPattern {
		totalAbund += isotope.Abundance
	}

	cutOffSum := totalAbund * (pctDiscardedAbund / 100.0)

	var cutOffAbund float64
	var idxCutOff int
	cutOffAbund = 0.0
	for idxCutOff = isoPattern.Length() - 1; cutOffAbund <= cutOffSum; idxCutOff-- {
		cutOffAbund += isoPattern[idxCutOff].Abundance
	}
	idxCutOff++

	isoPattern = isoPattern[0 : idxCutOff+1 : idxCutOff+1]
	order.QuickSort(isoPattern)
	*prtIsotPattern = isoPattern

}

//IosotpePatternCutOffThreshold cuts off by threshold: percentage vs the biggest peak.
func IosotpePatternCutOffThreshold(prtIsotPattern *chemistry.Isotopes, pctThreshVsBiggest float64) {
	isoPattern := *prtIsotPattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(chemistry.Isotopes)
		ans = asserted.CompareAbund(i, j) * (-1)
		return
	}

	order.PartialQuickSortFunc(isoPattern, 0, isoPattern.Length()-1, surrogateFuncAbundReverse)

	threshAbund := isoPattern[0].Abundance * (pctThreshVsBiggest / 100.0)

	var idxCutOff int
	for idxCutOff = isoPattern.Length() - 1; isoPattern[idxCutOff].Abundance <= threshAbund; idxCutOff-- {
	}
	idxCutOff++

	isoPattern = isoPattern[0 : idxCutOff+1 : idxCutOff+1]
	order.QuickSort(isoPattern)
	*prtIsotPattern = isoPattern
}
