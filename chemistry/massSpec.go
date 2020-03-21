package chemistry

import (
	"PainTheMaster/mybraly/order"
	"math"
)

//IsotopePatternCombination combines two isotope pattern
func IsotopePatternCombination(isotopePatternTo, isotopePatternFrom Isotopes) (isotopePattern Isotopes) {
	isotopePattern = isotopePatternTo

	tempPattern := make([]Isotopes, isotopePatternFrom.Length())
	for row := range tempPattern {
		tempPattern[row] = make(Isotopes, isotopePattern.Length())
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
func CompositionIsotopePattern(C ChemComposition) (isotopePattern Isotopes) {
	//	C[0].Elem
	isotopePattern = make(Isotopes, 1)
	isotopePattern[0].Mass = 0.0
	isotopePattern[0].Abundance = 1.0

	//Each composing element: count and sort
	for _, component := range C {
		componentCount := component.Count
		componentIsotopePatt := component.Elem.ChemCollection.IsotopePattern
		//Each atom of the element
		for coutAtom := 1; coutAtom <= componentCount; coutAtom++ {
			//Each newly added isotope
			isotopePattern = IsotopePatternCombination(isotopePattern, componentIsotopePatt)
		}
	}
	return isotopePattern
}

//IsotopePatternCutOffNum cuts off the IsotopePattern cuts off *prtIsotPattern by intensity.
//Only top num peaks survive.
func IsotopePatternCutOffNum(prtIsotopePattern *Isotopes, num int) {
	isoPattern := *prtIsotopePattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(Isotopes)
		ans = asserted.CompareAbund(i, j) * (-1)
		return
	}

	order.PartialQuickSortFunc(isoPattern, 0, isoPattern.Length()-1, surrogateFuncAbundReverse)

	isoPattern = isoPattern[0:num:num]

	order.QuickSort(isoPattern)

	*prtIsotopePattern = isoPattern
}

//IosotpePatternCutOffTotalAbund cuts off the by total abundance percentage.
//Maximum sum of discarded peaks is not more than discardedAbund
func IosotpePatternCutOffTotalAbund(prtIsotopePattern *Isotopes, pctDiscardedAbund float64) {
	isoPattern := *prtIsotopePattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(Isotopes)
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
	*prtIsotopePattern = isoPattern

}

//IosotpePatternCutOffThreshold cuts off by threshold: percentage vs the biggest peak.
func IosotpePatternCutOffThreshold(prtIsotopePattern *Isotopes, pctThreshVsBiggest float64) {
	isoPattern := *prtIsotopePattern

	surrogateFuncAbundReverse := func(tobeCompared order.Sorter, i int, j int) (ans int) {
		asserted := tobeCompared.(Isotopes)
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
	*prtIsotopePattern = isoPattern
}

//IsotopePatternNormalization nomarizes isotope pattern
func IsotopePatternNormalization(prtIsotopePattern *Isotopes) {
	isotoPattern := *prtIsotopePattern
	var abundTotal float64
	abundTotal = 0
	for _, isotope := range isotoPattern {
		abundTotal += isotope.Abundance
	}
	scale := 1.0 / abundTotal
	for _, isotope := range isotoPattern {
		isotope.Abundance *= scale
	}
}

//IsotopeRounding calculates rounded isotoper patterns
func IsotopeRounding(isotopePattern Isotopes) (roundedPattern Isotopes) {

	linearSearch := func(isotopes Isotopes, mass float64) (boolFound bool, idxFound int) {
		var isotopeCompare Isotope
		boolFound = false
		for idxFound, isotopeCompare = range isotopes {
			if isotopeCompare.Mass == mass {
				boolFound = true
				break
			}
		}
		return
	}

	binarySearch := func(isotopes Isotopes, mass float64) (boolFound bool, idxFound int) {
		const maxBinarySearchLength = 16

		order.QuickSort(isotopes)
		boolFound = false
		ini := 0
		end := isotopes.Length() - 1
		for {
			if isotopes.Length() <= maxBinarySearchLength {
				boolFound, idxFound = linearSearch(isotopes, mass)
				break
			} else {
				middle := ini + (end-ini)/2
				if isotopes[middle].Mass > mass {
					end = middle
				} else if isotopes[middle].Mass < mass {
					ini = middle
				} else {
					boolFound = true
					idxFound = middle
					break
				}
			}
		}
		return
	}

	for _, isotopeCompare := range isotopePattern {
		roundedMass := math.Round(isotopeCompare.Mass)
		boolFound, idxFound := binarySearch(roundedPattern, roundedMass)
		if boolFound {
			roundedPattern[idxFound].Abundance += isotopeCompare.Abundance
		} else {
			isotopeCompare.Mass = roundedMass
			roundedPattern = append(roundedPattern, isotopeCompare)
		}
	}
	return
}
