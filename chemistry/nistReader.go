package chemistry

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"os"
	"strconv"
)

//PeriodicTableMaker reads NIST's isotope file and make a PefiodicTable pt containing a list of stable isotopes
func PeriodicTableMaker(fileName string) (pt PeriodicTable) {
	nistFile, err := os.Open(fileName)
	defer nistFile.Close()
	if err != nil {
		fmt.Println("NIST file open error")
	}

	br := bufio.NewReader(nistFile)

	for err == nil {
		var elem Element
		var natural bool
		elem, natural, err = extractElement(br)
		if natural {
			boolFound, idxFound := peridicTableSearch(pt, elem)
			if boolFound {
				pt[idxFound].IsotopePattern = append(pt[idxFound].IsotopePattern, elem.IsotopePattern[0])
			} else {
				pt = append(pt, elem)
			}
		}
	}

	return
}

func extractElement(br *bufio.Reader) (elem Element, naturalIsotpope bool, err error) {
	var line []byte
	var done bool

	elem.IsotopePattern = make(Isotopes, 1)
	elem.IsotopePattern[0].Abundance = -1.0

	naturalIsotpope = false
	err = nil
	done = false
	for err == nil {
		line, _, err = br.ReadLine()
		if bytes.Contains(line, []byte("Atomic Number = ")) {
			elem.AtomNumber = extractAtomicNumber(line)
		} else if bytes.Contains(line, []byte("Atomic Symbol = ")) {
			elem.Symbol = extractAtomicSymbol(line)
		} else if bytes.Contains(line, []byte("Mass Number = ")) {
			elem.IsotopePattern[0].MassNumber = extractMassNumber(line)
		} else if bytes.Contains(line, []byte("Relative Atomic Mass = ")) {
			elem.IsotopePattern[0].Mass = extrractAtomicMass(line)
		} else if bytes.Contains(line, []byte("Isotopic Composition = ")) {
			elem.IsotopePattern[0].Abundance = extrractIsotopicComposition(line)
			if elem.IsotopePattern[0].Abundance > 0.0 {
				naturalIsotpope = true
			}
		} else if bytes.Contains(line, []byte("Standard Atomic Weight = ")) {
			elem.FormularWeight = extractAtomicWeight(line)
		} else if bytes.Contains(line, []byte("Notes = ")) {
			done = true
		} else if len(line) == 0 {
			if err == nil && !done {
				fmt.Println("extractElement: End of a section with somthing incomplete")
				fmt.Println("incomplete element:", elem)
				err = errors.New("extractElement error: end of section with missing data")
			}
			break
		}
	}
	return
}

func extractAtomicWeight(line []byte) (atomicWeight float64) {
	trimmed := bytes.Replace(line, []byte("Standard Atomic Weight = "), []byte(""), -1)
	lParen := bytes.Index(trimmed, []byte("("))
	if lParen >= 0 {
		trimmed = append(trimmed[0:lParen], trimmed[lParen+1:]...)
	}
	rParen := bytes.Index(trimmed, []byte(")"))
	if rParen >= 0 {
		trimmed = append(trimmed[0:rParen], trimmed[rParen+1:]...)
	}
	lSqParen := bytes.Index(trimmed, []byte("["))
	if lSqParen >= 0 {
		trimmed = append(trimmed[0:lSqParen], trimmed[lSqParen+1:]...)
	}
	rSqParen := bytes.Index(trimmed, []byte("]"))
	if rSqParen >= 0 {
		trimmed = trimmed[0:rSqParen]
		//		trimmed = append(trimmed[0:rSqParen], trimmed[rSqParen+1:]...)
	}

	calcAverage := func(intermdeiate []byte) (average float64) {
		idxComma := bytes.Index(intermdeiate, []byte(","))
		if idxComma >= 0 {
			leftBytes := intermdeiate[0:idxComma]
			rightBytes := intermdeiate[idxComma+1:]

			leftFloat, leftErr := strconv.ParseFloat(string(leftBytes), 64)
			if leftErr != nil {
				fmt.Println("extractAtomicWeight error", leftErr, ",", string(leftBytes))
			}
			rightFloat, rightErr := strconv.ParseFloat(string(leftBytes), 64)
			if rightErr != nil {
				fmt.Println("extractAtomicWeight error", rightErr, ",", string(rightBytes))
			}
			average = (leftFloat + rightFloat) / 2.0
		} else {
			singleFloat, singleErr := strconv.ParseFloat(string(intermdeiate), 64)
			if singleErr != nil {
				fmt.Println("extractAtomicWeight error", singleErr, ",", string(intermdeiate))
			}
			average = singleFloat
		}
		return
	}

	atomicWeight = calcAverage(trimmed)

	return
}

func extrractIsotopicComposition(line []byte) (isotopicComposition float64) {
	trimmed := bytes.Replace(line, []byte("Isotopic Composition = "), []byte(""), -1)
	lParen := bytes.Index(trimmed, []byte("("))
	if lParen >= 0 {
		trimmed = append(trimmed[0:lParen], trimmed[lParen+1:]...)
	}
	rParen := bytes.Index(trimmed, []byte(")"))
	if rParen >= 0 {
		trimmed = append(trimmed[0:rParen], trimmed[rParen+1:]...)
	}
	if len(trimmed) == 0 {
		isotopicComposition = -1.0
	} else {
		var err error
		isotopicComposition, err = strconv.ParseFloat(string(trimmed), 64)
		if err != nil {
			fmt.Println("extractIsotpoicComposition error:", err)
		}
	}
	return
}

func extrractAtomicMass(line []byte) (atomicMass float64) {
	trimmed := bytes.Replace(line, []byte("Relative Atomic Mass = "), []byte(""), -1)
	lParen := bytes.Index(trimmed, []byte("("))
	if lParen >= 0 {
		trimmed = append(trimmed[0:lParen], trimmed[lParen+1:]...)
	}
	rParen := bytes.Index(trimmed, []byte(")"))
	if rParen >= 0 {
		trimmed = append(trimmed[0:rParen], trimmed[rParen+1:]...)
	}
	sharp := bytes.Index(trimmed, []byte("#"))
	if sharp >= 0 {
		trimmed = append(trimmed[0:sharp], trimmed[sharp+1:]...)
	}

	var err error
	atomicMass, err = strconv.ParseFloat(string(trimmed), 64)
	if err != nil {
		fmt.Println("extractAtomicMass error:", err)
	}
	return
}

func extractMassNumber(line []byte) (massNumber int) {
	trimmed := bytes.Replace(line, []byte("Mass Number = "), []byte(""), -1)
	var err error
	massNumber, err = strconv.Atoi(string(trimmed))
	if err != nil {
		fmt.Println("extractMassNumber error:", err)
	}
	return
}

func extractAtomicSymbol(line []byte) (atomicSymbol string) {
	fmt.Println("atomic symbol line:", string(line))
	trimmed := bytes.Replace(line, []byte("Atomic Symbol = "), []byte(""), -1)
	atomicSymbol = string(trimmed)
	if atomicSymbol == "D" || atomicSymbol == "T" {
		atomicSymbol = "H"
	}
	return
}

func extractAtomicNumber(line []byte) (atomicNumber int) {
	trimmed := bytes.Replace(line, []byte("Atomic Number = "), []byte(""), -1)
	var err error
	atomicNumber, err = strconv.Atoi(string(trimmed))
	if err != nil {
		fmt.Println("extractAtomicNumber error:", err)
	}
	return
}

func peridicTableSearch(pt PeriodicTable, elem Element) (boolFound bool, idxFound int) {
	boolFound = false
	idxFound = -1

	for idxSearch := pt.Length() - 1; idxSearch >= 0; idxSearch-- {
		if pt[idxSearch].AtomNumber == elem.AtomNumber {
			boolFound = true
			idxFound = idxSearch
			break
		}
	}
	return
}
