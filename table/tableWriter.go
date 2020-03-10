package table

import (
	"bufio"
	"os"
	"strconv"
)

// WriteTable writes a table
func WriteTable(matrix [][]float64, file *os.File) {

	writer := bufio.NewWriter(file)

	rows := len(matrix)
	for i := 0; i <= rows-1; i++ {
		cols := len(matrix[i])
		for j := 0; j <= cols-2; j++ {
			writer.WriteString(strconv.FormatFloat(matrix[i][j], 'e', 10, 64))
			writer.WriteByte(',')
		}
		{
			j := cols - 1
			writer.WriteString(strconv.FormatFloat(matrix[i][j], 'e', 10, 64))
		}
		writer.WriteString("\n")
		writer.Flush()
	}

}
