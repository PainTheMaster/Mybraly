package mnist

import (
	"PainTheMaster/mybraly/math/linearalgebra"
	"image"
	"image/color"
	"image/png"
	"os"
)

//GetImage gets bitmap matrix of dataNum-th data in the given file and put it out as "ans" matrix.
func GetImage(file *os.File, dataNum int) (ans [][]uint8) {
	const (
		headerOffset = 16 //bytes
		cols         = 28 //pixels, bytes
		rows         = 28 //pixels, bytes
	)

	ans = make([][]uint8, rows)
	for i := range ans {
		ans[i] = make([]uint8, cols)
	}

	dataOffset := headerOffset + cols*rows*dataNum

	buf := make([]uint8, cols*rows)
	file.ReadAt(buf, int64(dataOffset))
	for i := 0; i <= rows-1; i++ {
		for j := 0; j <= cols-1; j++ {
			ans[i][j] = buf[cols*i+j]
		}
	}

	return ans
}

//ImagToColvec transforms [][]uint8 image to a linearalgebra.ColImagToColvec type column vector.
func ImagToColvec(imag [][]uint8) (colvec linearalgebra.Colvec) {
	rows := len(imag)
	cols := len(imag[0])
	colvec = make(linearalgebra.Colvec, cols*rows)
	for i := range imag {
		for j := range imag[i] {
			colvec[cols*i+j] = float64(imag[i][j])
		}
	}
	return
}

//GetLabel gets dataNum-th label from a given file
func GetLabel(file *os.File, dataNum int) (ans int) {
	const (
		headerOffset = 8
		dataSize     = 1 //byte
	)

	dataOffset := headerOffset + dataNum
	buf := make([]uint8, dataSize)
	file.ReadAt(buf, int64(dataOffset))

	ans = int(buf[0])

	return
}

//LabelOneHot makes one-hot expression of the label
func LabelOneHot(label int) (colvec linearalgebra.Colvec) {
	const lenVec = 10
	colvec = make(linearalgebra.Colvec, lenVec)
	colvec[label] = 1.0
	return
}

//GetItemNumber gets the number of items contained in the file
func GetItemNumber(file *os.File) (ans int) {
	const (
		headerOffset = 4
		dataSize     = 4 //bytes
		byteToBits   = 8
	)

	buf := make([]uint8, dataSize)
	file.ReadAt(buf, int64(headerOffset))

	ans = 0
	for i := range buf {
		ans += int(buf[i]) << ((dataSize - i - 1) * byteToBits)
	}
	return
}

//ToImage makes image of a bitmap matrix "cell". The output is a png file
func ToImage(cells [][]uint8, file *os.File) {

	rows := len(cells)
	cols := len(cells[0])

	var cellLen int = 10 // pixels

	rectangle := image.Rect(0, 0, cellLen*cols, cellLen*rows)
	nrgba := image.NewNRGBA(rectangle)

	var grayScale color.Gray
	var cellNRBGA color.NRGBA

	for i := 0; i <= rows-1; i++ {
		for j := 0; j <= cols-1; j++ {
			grayScale.Y = cells[i][j]

			r, g, b, a := grayScale.RGBA()
			cellNRBGA.R = uint8(r)
			cellNRBGA.G = uint8(g)
			cellNRBGA.B = uint8(b)
			cellNRBGA.A = uint8(a)

			ystart := cellLen * i
			xstart := cellLen * j

			for dy := 0; dy <= cellLen-1; dy++ {
				for dx := 0; dx <= cellLen-1; dx++ {
					nrgba.SetNRGBA(xstart+dx, ystart+dy, cellNRBGA)
				}
			}
		}
	}

	png.Encode(file, nrgba)
	file.Close()

}
