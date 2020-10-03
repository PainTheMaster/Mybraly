package mass

import (
	"PainTheMaster/mybraly/mymath/round"
	"PainTheMaster/mybraly/stat"
	"math"
	"math/rand"
)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////     Dummy signal     //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Signal generates mass signals
func Signal(mass float64, pksInClust, chFrom, chTo int, intens float64) (sig [][]float64) {
	const dimInClust = 0.6
	const dimInterClust = 0.3
	const addMass = 1.0
	const sigmaMass = 0.01 / 3.0
	numCharg := chTo - chFrom + 1
	chargeSlice := make([]float64, numCharg)
	for i := range chargeSlice {
		chargeSlice[i] = float64(chFrom + i)
	}

	sig = make([][]float64, numCharg*pksInClust*2)
	for i := range sig {
		sig[i] = make([]float64, 2)
	}

	//rand.NormFloat64()
	for idx, charge := range chargeSlice {
		for peak := 0; peak <= pksInClust-1; peak++ {
			sig[idx*pksInClust+peak][0] = (mass/charge + addMass) + addMass/charge*float64(peak) + rand.NormFloat64()*rand.NormFloat64()*sigmaMass
			sig[idx*pksInClust+peak][1] = intens * math.Pow(dimInterClust, float64(idx)) * math.Pow(dimInClust, float64(peak))
			sig[(idx+1)*pksInClust+peak][0] = (mass/charge + addMass) + addMass/charge*float64(peak) + addMass/charge*0.5
			sig[(idx+1)*pksInClust+2*peak][1] = 0.0
		}
	}
	return
}

//Noize generates noizes in massspectrum.
func Noize(from, to, intens float64, pks int) (sig [][]float64) {
	const sigma = 0.5 / 0.68
	sig = make([][]float64, pks)
	for i := range sig {
		sig[i] = make([]float64, 2)
	}

	width := to - from
	for i := range sig {
		sig[i][0] = from + width*rand.Float64()
		sig[i][1] = intens * (1.0 + rand.NormFloat64()*sigma)
	}
	return
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////  Signal for fitting  //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
const (
	MH         = 1.00782503223
	MN         = 2.01410177812 - 1.00782503223 //Neutron
	Resolution = 0.001

//	CutOffRatio = 0.0001
)

//ModelSpecies is a struct to define a species that appears on the spectrum.
type ModelSpecies struct {
	IntensSpcs  float64   //Intensity of the spieces.
	MonoIsoMass float64   //MonoIsotopic mass. Zero charge.
	NumIstope   float64   //Number of isotopes. "n" in binomial distribution.
	IsotopRatio float64   //ratio of heavier isottope. "p" in binomial distribution.
	IntensChg   []float64 //intensities of each charges.
}

//ModelSpectr is a set of parameters to generate a spectrum and the spectrum itself
type ModelSpectr struct {
	Resolution  float64        //interval between data points
	Sigma       float64        //sigma of bell curb beaks
	PeakWidth   float64        //sigma * peakWidth is the calculation range of each peak.
	CutOffRatio float64        //peaks weaker than this is not calculated
	Species     []ModelSpecies //Slice of contributing species.
	Spectr      []float64      //resulting mass spectra.
}

//GenerSpectr generates a spectrum
func (modelSpectr *ModelSpectr) GenerSpectr() {
	spectr := make([]float64, 1)
	for iSpc := range modelSpectr.Species {
		spcFocus := modelSpectr.Species[iSpc]
		for chg := 1; chg <= len(spcFocus.IntensChg)-1; chg++ {
			offset, specTemp := modelSpectr.partialSpec(spcFocus, chg)
			spectr = integrate(spectr, offset, specTemp)
		}
	}
	modelSpectr.Spectr = spectr
}

//integrate combines peaks of multiple charges.
func integrate(a []float64, offset int, b []float64) (ans []float64) {
	origLen := len(a)
	requLen := offset + len(b)
	if origLen < requLen {
		if offset <= origLen-1 {
			for i := 0; offset+i <= origLen-1; i++ {
				a[offset+i] += b[i]
			}
			a = append(a, b[origLen-offset:]...)
		} else { //if offset is righter than the end of a
			a = append(a, (make([]float64, offset-origLen))...)
			a = append(a, b...)
		}
	} else {
		for i := range b {
			a[offset+i] += b[i]
		}
	}
	ans = a
	return
}

//partialSpec calculates contribution of one ModelSpecies modelSpc with designated charge chg.
func (modelSpectr ModelSpectr) partialSpec(modelSpc ModelSpecies, chg int) (offset int, spec []float64) {
	intensSpcs := modelSpc.IntensSpcs
	intensChg := modelSpc.IntensChg[chg]
	reso := modelSpectr.Resolution
	sigma := modelSpectr.Sigma
	tail := sigma * modelSpectr.PeakWidth //single peak cut off range in m/z domain

	monoIsoMPerZ := modelSpc.MonoIsoMass/float64(chg) + MH
	leftMPerZ := monoIsoMPerZ - tail
	offset = int(round.Round(leftMPerZ / reso))

	numPks := rangeFinder(modelSpc, modelSpectr.CutOffRatio)
	rightMPerZ := (modelSpc.MonoIsoMass+MN*float64(numPks-1))/float64(chg) + MH + tail
	length := int(round.Round((rightMPerZ-leftMPerZ)/reso)) + 1

	spec = make([]float64, length)

	relIdxWidth := int(tail / reso)
	bellCurb := make([]float64, relIdxWidth*2+1)
	for i := 0; i <= relIdxWidth*2; i++ {
		bellCurb[i] = math.Exp(-1.0 * math.Pow(reso*float64(i-relIdxWidth), 2.0) / (2.0 * math.Pow(sigma, 2.0)))
	}

	n := modelSpc.NumIstope
	p := modelSpc.IsotopRatio
	for peakIdx := 0; peakIdx <= numPks-1; peakIdx++ {
		peakIntens := stat.Binomial64(n, float64(peakIdx), p)

		//		focusCentrMPerZ := monoIsoMPerZ + MN*float64(peakIdx)
		//		focusCentrIdx := int(round.Round((focusCentrMPerZ - monoIsoMPerZ) / reso))

		focusCentrIdx := int(round.Round((monoIsoMPerZ + MN*float64(peakIdx)/float64(chg) - leftMPerZ) / reso))
		focusBeginIdx := focusCentrIdx - relIdxWidth
		for relIdx := 0; relIdx <= 2*relIdxWidth; relIdx++ {
			spec[focusBeginIdx+relIdx] = intensSpcs * intensChg * peakIntens * bellCurb[relIdx]
		}
	}
	return
}

//rangeFinder calculates the number of peaks of an instance of ModelSpecies ms as per cut-off ratio
func rangeFinder(ms ModelSpecies, cutOffRatio float64) (numPks int) {
	n64 := ms.NumIstope
	nInt := int(n64)
	p := ms.IsotopRatio

	var idxMax int
	var peak float64
	for idxMax = 0; ; idxMax++ {
		if idxMax == nInt-1 {
			break
		}
		current := stat.Binomial64(n64, float64(idxMax), p)
		next := stat.Binomial64(n64, float64(idxMax)+1.0, p)
		if current > next {
			peak = current
			break
		}
	}

	idxCutoff := idxMax + 1
	for ; ; idxCutoff++ {
		if idxCutoff == nInt-1 {
			break
		}
		next := stat.Binomial64(n64, float64(idxCutoff), p)
		if next < peak*cutOffRatio {
			break
		}
	}
	numPks = idxCutoff + 1
	return
}
