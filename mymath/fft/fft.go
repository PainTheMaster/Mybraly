package fft

import (
	"fmt"
	"math"
)

var wRev, wForw [][]complex128

// FFT performs fast fourier transformation
func FFT(f []float64) (F []complex128) {

	zeroFill(&f)
	bitWidth := uint(bitWidth(len(f) - 1))
	prepOmega(1 << (bitWidth))

	shuffle := make([]int, len(f))
	for i := range shuffle {
		shuffle[i] = bitreverse(i, int(bitWidth))
	}

	buf := make([][]complex128, bitWidth+1)
	for i := range buf {
		buf[i] = make([]complex128, len(f))
	}

	var stage uint
	stage = 0
	for i := 0; i <= len(f)-1; i++ {
		convIdx := shuffle[i]
		buf[0][i] = complex(f[convIdx], 0.0)
	}

	stage = 1
	cellSize := (1 << stage)
	for stage <= bitWidth {
		halfCellSize := cellSize >> 1
		for col := 0; col+cellSize-1 <= len(f)-1; col += cellSize {
			for k := 0; k <= halfCellSize-1; k++ {
				buf[stage][col+k] = wRev[stage][k] * buf[stage-1][col+k+halfCellSize]
				buf[stage][col+k+halfCellSize] = -wRev[stage][k] * buf[stage-1][col+k+halfCellSize]

				buf[stage][col+k] += buf[stage-1][col+k]
				buf[stage][col+k+halfCellSize] += buf[stage-1][col+k]
			}
		}
		stage++
		cellSize = (1 << stage)
	}
	stage--

	F = buf[stage]

	for i := range F {
		F[i] /= complex(float64(len(F)), 0.0)
	}

	return
}

func rft(F []complex128) (f []complex128) {

	size := len(F)
	f = make([]complex128, size)

	rot := 2.0 * math.Pi / float64(size)

	var temp complex128
	for l := 0; l <= size-1; l++ {
		temp = 0 + 0i
		for k := 0; k <= size-1; k++ {
			omega := complex(math.Cos(rot*float64(k*l)), math.Sin(rot*float64(k*l)))
			temp += omega * F[k]
		}
		f[l] = temp
	}

	return
}

func ft(f []float64) (F []complex128) {

	size := len(f)
	F = make([]complex128, size)

	rot := 2.0 * math.Pi / float64(size)

	var temp complex128
	for l := 0; l <= size-1; l++ {
		temp = 0 + 0i
		for k := 0; k <= size-1; k++ {
			omega := complex(math.Cos(rot*float64(k*l)), -math.Sin(rot*float64(k*l)))
			temp += omega * complex(f[k], 0.0)
		}
		F[l] = temp / complex(float64(size), 0.0)
	}

	return
}

func zeroFill(f *[]float64) {
	var origBitWidth, newBitWidth uint
	var newLength int
	var temp []float64

	/*	for ; (len(*f) >> origBitWidth) > 0; origBitWidth++ {
		}
	*/
	origBitWidth = uint(bitWidth(len(*f)))

	if len(*f) == (1 << (origBitWidth - 1)) {
	} else {
		newBitWidth = origBitWidth + 1
		newLength = 1 << (newBitWidth - 1)
		temp = make([]float64, newLength-len(*f))
		*f = append(*f, temp...)
	}
}

//Bitreverse is a function that rearranges the index for FFT
func bitreverse(x int, bitWidth int) int {
	//	ux := uint32(x)
	var k uint
	var uBitWidth = uint(bitWidth)
	var temp int

	for k = 0; k <= uBitWidth-1; k++ {
		temp |= (x & 1) << (uBitWidth - k - 1)
		x = x >> 1
	}

	return temp
}

//prepOmega prepares table for wRev and wForw
//argument "size" is literally the "size", normally 1, 2, 4, 8..., and NOT a logarithm of them
//wForw[i] corresponds to a series of 2^i th root of unity, and contains 2^i datum (w^0 to w^(i-1))
func prepOmega(size int) {

	var numBitSize int

	/*	for ; (size >> uint(numBitSize)) > 0; numBitSize++ {
		}*/

	numBitSize = bitWidth(size)

	wForw = make([][]complex128, numBitSize)
	wRev = make([][]complex128, numBitSize)

	for i := 0; 1<<uint(i) <= size; i++ {

		thisDivision := 1 << uint(i)
		wForw[i] = make([]complex128, thisDivision)
		wRev[i] = make([]complex128, thisDivision)

		unitRad := 2.0 * math.Pi / float64(thisDivision)
		for j := 0; j <= thisDivision-1; j++ {
			wForw[i][j] = complex(math.Cos(unitRad*float64(j)), math.Sin(unitRad*float64(j)))
			wRev[i][j] = complex(math.Cos(unitRad*float64(j)), -math.Sin(unitRad*float64(j)))
		}
	}
}

//bitWidth returns bit width of an integer x. ex) bitwidth(4)==3
func bitWidth(x int) int {
	var width uint
	for width = 0; (x >> width) > 0; width++ {
	}

	return int(width)
}

//Test is an exported test field
func Test() {
	div := 64
	f := make([]float64, div)
	for i := 0; i <= div-1; i++ {
		f[i] = math.Sin(2.0 * math.Pi / float64(div) * float64(i))
	}

	F := FFT(f)
	//	sF := ft(f)

	for i := range F {
		fmt.Printf("%d,%f\n", i, F[i])
	}
	fmt.Println()

	for i := range F {
		fmt.Printf("%d,%f\n", i, cabs(F[i]))
	}
	fmt.Println()

	regen := rft(F)
	for i := range regen {
		fmt.Printf("%d,%f\n", i, real(regen[i]))
	}
}

func cabs(z complex128) float64 {
	temp := math.Pow(real(z), 2.0) + math.Pow(imag(z), 2.0)
	return math.Sqrt(temp)
}

/*func Test() {
	//	f8, f10 := make([]float64, 8), make([]float64, 10)

	f8 := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
	f10 := []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}

	fmt.Println("len(f8)", len(f8))
	fmt.Println("len(f10)", len(f10))

	zeroFill(&f8)
	zeroFill(&f10)
	fmt.Println("zerofilled")
	fmt.Println("len(f8)", len(f8))
	fmt.Println("len(f10)", len(f10))

}*/

/*func Test() {
	size := 8

	prepOmega(size)

	fmt.Println("len(wForw):", len(wForw))

	for i := 0; i <= len(wForw)-1; i++ {
		fmt.Println("i:", i)
		fmt.Println("wForw len:", len(wForw[i]), ", wRev len:", len(wRev[i]))
		for j := 0; j <= len(wForw[i])-1; j++ {
			fmt.Println(wForw[i][j], ", ", wRev[i][j])
		}
	}
}*/
