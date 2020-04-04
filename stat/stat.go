package stat

import "math"

//TwoTailSigma returns scale of sigma (standard deviation) that corresponds to thee percentage of two tail testing of normal distribution
func TwoTailSigma(pct float64) (nSigma float64) {
	const (
		step95      = 5
		step99      = 1
		step100     = 0.1
		smallNumber = 0.001
	)

	if pct <= 95.0 {
		idxFloor := int(pct / step95)
		idxSealing := idxFloor + 1

		valFloor := tabNormDistrib95[idxFloor]
		valSealing := tabNormDistrib95[idxSealing]

		pctFloor := float64(idxFloor) * step95

		nSigma = valFloor + (valSealing-valFloor)*(pct-pctFloor)/step95
	} else if pct <= 99.0 {
		deltaFrom95 := pct - 95.0
		idxFloor := int(deltaFrom95 / step99)
		idxSealing := idxFloor + 1

		valFloor := tabNormDistrib99[idxFloor]
		valSealing := tabNormDistrib99[idxSealing]

		pctFloor := 95.0 + float64(idxFloor)*step99

		nSigma = valFloor + (valSealing-valFloor)*(pct-pctFloor)/step99
	} else if pct <= 99.9 {
		deltaFrom99 := pct - 99.0
		idxFloor := int(deltaFrom99 / step100)
		idxSealing := idxFloor + 1

		valFloor := tabNormDistrib100[idxFloor]
		valSealing := tabNormDistrib100[idxSealing]

		pctFloor := 99.0 + float64(idxFloor)*step100

		nSigma = valFloor + (valSealing-valFloor)*(pct-pctFloor)/step100
	} else {
		nSigma = math.Inf(1)
	}
	return
}
