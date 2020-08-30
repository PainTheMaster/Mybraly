package round

import "math"

//Round rounds x to 0 decimal place.
func Round(x float64) (ans float64) {
	ans = math.Floor(x + 0.5)
	return
}

//RoundN rouds x to n decimal place
func RoundN(x float64, n int) (ans float64) {
	shifter := math.Pow(10, float64(n))
	x *= shifter
	x += 0.5
	x = math.Floor(x)
	ans = x / shifter
	return
}
