package mnist

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
)

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
	fmt.Println("len of buf:", len(buf))
	for i := 0; i <= rows-1; i++ {
		for j := 0; j <= cols-1; j++ {
			ans[i][j] = buf[cols*i+j]
		}
	}

	return ans
}

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

}
