package order

import "PainTheMaster/mybraly/service"

const (
	maxHeapLength = 16
)

type term struct {
	ini int
	end int
}

//QuickSort sorts sorter Sorter by using default magnitude function DefaultMagnitude
func QuickSort(sorter Sorter) {
	defaultCompare := func(sorter Sorter, i int, j int) int {
		return sorter.Compare(i, j)
	}
	PartialQuickSortFunc(sorter, 0, sorter.Length()-1, defaultCompare)
}

//QuickSortReverse sorts sorter Sorter by using default magnitude function DefaultMagnitude
func QuickSortReverse(sorter Sorter) {
	defaultCompare := func(sorter Sorter, i int, j int) int {
		return sorter.Compare(i, j) * (-1)
	}
	PartialQuickSortFunc(sorter, 0, sorter.Length()-1, defaultCompare)
}

//PartialQuickSort sorts sorter Sorter by using default magnitude function DefaultMagnitude
func PartialQuickSort(sorter Sorter, ini int, end int) {
	defaultCompare := func(sorter Sorter, i int, j int) int {
		return sorter.Compare(i, j)
	}
	PartialQuickSortFunc(sorter, ini, end, defaultCompare)
}

//PartialQuickSortReverse reversely sorts sorter Sorter by using default magnitude function DefaultMagnitude
func PartialQuickSortReverse(sorter Sorter, ini int, end int) {
	defaultCompare := func(sorter Sorter, i int, j int) int {
		return sorter.Compare(i, j) * (-1)
	}
	PartialQuickSortFunc(sorter, ini, end, defaultCompare)
}

// PartialQuickSortFunc sorts sorter Sorter by quick sort method
func PartialQuickSortFunc(sorter Sorter, ini int, end int, compare func(Sorter, int, int) int) {
	var stack service.Stack
	var span term

	span.ini = ini
	span.end = end
	stack.Push(span)

	for stack.Depth() >= 1 {
		span = stack.Pop().(term)
		pivot := unitQuickSort(sorter, span.ini, span.end, compare)

		//left
		if pivot-span.ini+1 > maxHeapLength {
			left := term{ini: span.ini, end: pivot}
			stack.Push(left)
		} else {
			PartialHeapSortFunc(sorter, span.ini, pivot, compare)
		}

		//right
		if span.end-pivot > maxHeapLength {
			right := term{ini: pivot + 1, end: span.end}
			stack.Push(right)
		} else {
			PartialHeapSortFunc(sorter, pivot+1, span.end, compare)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////        Helper fuctions from here    ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

func unitQuickSort(sorter Sorter, ini int, end int, compare func(Sorter, int, int) int) int {

	//	startingIdxPiv, magPiv := pivoting(sorter, ini, end, magnitude)

	idxPiv := pivoting(sorter, ini, end, compare)

	frontL, frontR := ini, end

	for {
		//start moving frontL from the left to find the first bigger-than- or equal-to-pivot value
		for ; frontL <= idxPiv-1; frontL++ {
			if compare(sorter, frontL, idxPiv) >= 0 {
				break
			}
		}
		//start moving frontR from the right to find the first smaller-than-pivot value
		for ; frontR >= idxPiv+1; frontR-- {
			if compare(sorter, idxPiv, frontR) > 0 {
				break
			}
		}
		if frontL < frontR {
			sorter.Swap(frontL, frontR)
			if frontL == idxPiv {
				frontL++
				idxPiv = frontR
			} else if idxPiv == frontR {
				frontR--
				idxPiv = frontL
			}
		} else { //end of sort
			break
		}
	}

	return idxPiv
}

func pivoting(sorter Sorter, ini int, end int, compare func(Sorter, int, int) int) (idxPvt int) {
	a := ini
	b := ini + (end-ini)/2
	c := end

	AbiggerB := compare(sorter, a, b)
	AbiggerC := compare(sorter, a, c)
	BbiggerC := compare(sorter, b, c)

	if AbiggerB >= 0 && AbiggerC >= 0 {
		if BbiggerC >= 0 {
			idxPvt = b
		} else {
			idxPvt = c
		}
	} else if AbiggerB <= 0 && BbiggerC >= 0 {
		if AbiggerC >= 0 {
			idxPvt = a
		} else {
			idxPvt = c
		}
	} else if AbiggerC <= 0 && BbiggerC <= 0 {
		if AbiggerB >= 0 {
			idxPvt = a
		} else {
			idxPvt = b
		}
	}
	return
}
