package specialfunc

import (
	"math"
)

const (
	lnSqrt2Pi = 0.91893853320467266
	g         = 7
)

var coef = [...]float64{
	0.99999999999980993227684700473478,
	676.520368121885098567009190444019,
	-1259.13921672240287047156078755283,
	771.3234287776530788486528258894,
	-176.61502916214059906584551354,
	12.507343278686904814458936853,
	-0.13857109526572011689554707,
	9.984369578019570859563e-6,
	1.50563273514931155834e-7}

//LnGamma is logharithm of Lanchoz approximate of gamma function.
func LnGamma(z float64) (lng float64) {
	z -= 1.0

	base := float64(z+g) + 0.5

	lenCoef := len(coef)
	ser := coef[0]
	for i := 1; i <= lenCoef-1; i++ {
		ser += coef[i] / (z + float64(i))
	}

	lng = (z+0.5)*math.Log(base) - base + lnSqrt2Pi + math.Log(ser)
	return
}

//Gamma is gammafunction. This is function approximates gamma function with Lanchoz method.
func Gamma(z float64) (ans float64) {
	ans = math.Exp(LnGamma(z))
	return
}
