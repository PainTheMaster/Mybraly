package mass

import (
	"math"
	"math/rand"
)

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
