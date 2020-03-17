package table

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
)

// ReadTableFloat writes a table of float64
func ReadTableFloat(reader io.Reader, delim byte) (matrix [][]float64) {

	var currentLine int

	bufIn := bufio.NewReader(reader)

	currentLine = 0
	lineBuf, _, errReadLine := bufIn.ReadLine()
	if errReadLine != nil {
		fmt.Println("Header read error:", errReadLine)
	}

	countDelim := func(b []byte) int {
		length := len(b)
		var delimNum int
		for i := 0; i <= length-1; i++ {
			if b[i] == delim {
				delimNum++
			} else if (b[i] == '"') || (b[i] == '\'') {
				quaTemp := b[i]
				for i++; b[i] != quaTemp; i++ {
				}
				i++
				if i == length {
					return delimNum + 1
				}
			}
		}
		return delimNum + 1
	}

	cols := countDelim(lineBuf)

	const bufLenMat int = 100
	matrix = make([][]float64, bufLenMat)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}

	tokenizer := func(b []byte, data []float64) {
		length := len(b)
		var col, currentPos, nextDelim int

		col = 0
		for currentPos = 0; currentPos <= length-1; currentPos = nextDelim + 1 {
			//次の区切り文字を探す。
			for nextDelim = currentPos; nextDelim <= length-1 && b[nextDelim] != delim; nextDelim++ {
				//引用符号があった場合は閉じるインデックスまで進める。
				if b[nextDelim] == '\'' || b[nextDelim] == '"' {
					quaTemp := b[nextDelim]
					for nextDelim++; b[nextDelim] != quaTemp; nextDelim++ {
					}
					//このifを抜けた時点でnextDelimは終わりの引用符号を指している
				}
				//このforを抜けた時点でnextDelimは次の区切り文字上もしくは行の終端を１つオーバーランしたところを指している
			}
			var errParseFloat error
			data[col], errParseFloat = strconv.ParseFloat(string(b[currentPos:nextDelim]), 64)
			if errParseFloat != nil {
				fmt.Println("String to float64 conversion error:", errParseFloat)
			}
			col++
		}
	}

	if errReadLine == nil {
		tokenizer(lineBuf, matrix[currentLine])
	}

	for currentLine++; ; currentLine++ {
		lineBuf, _, errReadLine = bufIn.ReadLine()
		if errReadLine == nil {
			tokenizer(lineBuf, matrix[currentLine])
		} else {
			break
		}
		if currentLine%bufLenMat == bufLenMat-1 {
			bufExpand := make([][]float64, bufLenMat)
			for i := range matrix {
				bufExpand[i] = make([]float64, cols)
			}
			matrix = append(matrix, bufExpand...)
		}
	}

	matrix = matrix[:currentLine]

	return matrix

}

// ReadTableString reads a table of strings
func ReadTableString(reader io.Reader, delim byte) (table [][]string) {

	var currentLine int

	bufIn := bufio.NewReader(reader)

	currentLine = 0
	lineBuf, _, errReadLine := bufIn.ReadLine()
	if errReadLine != nil {
		fmt.Println("Header read error:", errReadLine)
	}

	countDelim := func(b []byte) int {
		length := len(b)
		var delimNum int
		for i := 0; i <= length-1; i++ {
			if b[i] == delim {
				delimNum++
			} else if (b[i] == '"') || (b[i] == '\'') {
				quaTemp := b[i]
				for i++; b[i] != quaTemp; i++ {
				}
				i++
				if i == length {
					return delimNum + 1
				}
			}
		}
		return delimNum + 1
	}

	cols := countDelim(lineBuf)

	const bufLenMat int = 100
	table = make([][]string, bufLenMat)
	for i := range table {
		table[i] = make([]string, cols)
	}

	tokenizer := func(b []byte, data []string) {
		length := len(b)
		var col, currentPos, nextDelim int

		col = 0
		for currentPos = 0; currentPos <= length-1; currentPos = nextDelim + 1 {
			//次の区切り文字を探す。
			for nextDelim = currentPos; nextDelim <= length-1 && b[nextDelim] != delim; nextDelim++ {
				//引用符号があった場合は閉じるインデックスまで進める。
				if b[nextDelim] == '\'' || b[nextDelim] == '"' {
					quaTemp := b[nextDelim]
					for nextDelim++; b[nextDelim] != quaTemp; nextDelim++ {
					}
					//このifを抜けた時点でnextDelimは終わりの引用符号を指している
				}
				//このforを抜けた時点でnextDelimは次の区切り文字上もしくは行の終端を１つオーバーランしたところを指している
			}
			data[col] = string(b[currentPos:nextDelim])

			col++
		}
	}

	if errReadLine == nil {
		tokenizer(lineBuf, table[currentLine])
	}

	for currentLine++; ; currentLine++ {
		lineBuf, _, errReadLine = bufIn.ReadLine()
		if errReadLine == nil {
			tokenizer(lineBuf, table[currentLine])
		} else {
			break
		}
		if currentLine%bufLenMat == bufLenMat-1 {
			bufExpand := make([][]string, bufLenMat)
			for i := range table {
				bufExpand[i] = make([]string, cols)
			}
			table = append(table, bufExpand...)
		}
	}

	table = table[:currentLine]

	return table

}

// ReadTableHeader reads a table of strings
func ReadTableHeader(reader io.ReadSeeker, delim byte) (header []string) {

	bufIn := bufio.NewReader(reader)

	lineBuf, _, errReadLine := bufIn.ReadLine()
	if errReadLine != nil {
		fmt.Println("Header read error:", errReadLine)
	}

	countDelim := func(b []byte) int {
		length := len(b)
		var delimNum int
		for i := 0; i <= length-1; i++ {
			if b[i] == delim {
				delimNum++
			} else if (b[i] == '"') || (b[i] == '\'') {
				quatatTemp := b[i]
				for i++; b[i] != quatatTemp; i++ {
				}
				i++
				if i == length {
					return delimNum + 1
				}
			}
		}
		return delimNum + 1
	}

	cols := countDelim(lineBuf)

	header = make([]string, cols)

	tokenizer := func(b []byte, data []string) {
		length := len(b)
		var col, currentPos, nextDelim int

		col = 0
		for currentPos = 0; currentPos <= length-1; currentPos = nextDelim + 1 {
			//次の区切り文字を探す。
			for nextDelim = currentPos; nextDelim <= length-1 && b[nextDelim] != delim; nextDelim++ {
				//引用符号があった場合は閉じるインデックスまで進める。
				if b[nextDelim] == '\'' || b[nextDelim] == '"' {
					quaTemp := b[nextDelim]
					for nextDelim++; b[nextDelim] != quaTemp; nextDelim++ {
					}
					//このifを抜けた時点でnextDelimは終わりの引用符号を指している
				}
				//このforを抜けた時点でnextDelimは次の区切り文字上もしくは行の終端を１つオーバーランしたところを指している
			}
			data[col] = string(b[currentPos:nextDelim])

			col++
		}
	}

	if errReadLine == nil {
		tokenizer(lineBuf, header)
		reader.Seek(int64(len(lineBuf)+1), io.SeekStart)
	}

	return header
}
