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
	magnitude := func(sorter Sorter, idx int) float64 {
		return sorter.DefaultMagnitude(idx)
	}
	PartialQuickSortFunc(sorter, 0, sorter.Length()-1, magnitude)
}

//QuickSortReverse sorts sorter Sorter by using default magnitude function DefaultMagnitude
func QuickSortReverse(sorter Sorter) {
	magnitude := func(sorter Sorter, idx int) float64 {
		return sorter.DefaultMagnitude(idx) * (-1.0)
	}
	PartialQuickSortFunc(sorter, 0, sorter.Length()-1, magnitude)
}

//PartialQuickSort sorts sorter Sorter by using default magnitude function DefaultMagnitude
func PartialQuickSort(sorter Sorter, ini int, end int) {
	magnitude := func(sorter Sorter, idx int) float64 {
		return sorter.DefaultMagnitude(idx)
	}
	PartialQuickSortFunc(sorter, ini, end, magnitude)
}

//PartialQuickSortReverse reversely sorts sorter Sorter by using default magnitude function DefaultMagnitude
func PartialQuickSortReverse(sorter Sorter, ini int, end int) {
	magnitude := func(sorter Sorter, idx int) float64 {
		return sorter.DefaultMagnitude(idx) * (-1.0)
	}
	PartialQuickSortFunc(sorter, ini, end, magnitude)
}

// PartialQuickSortFunc sorts sorter Sorter by quick sort method
func PartialQuickSortFunc(sorter Sorter, ini int, end int, magnitude func(Sorter, int) float64) {
	var stack service.Stack
	var span term

	span.ini = ini
	span.end = end
	stack.Push(span)

	for stack.Depth() >= 1 {
		span = stack.Pop().(term)
		pivot := unitQuickSort(sorter, span.ini, span.end, magnitude)

		//left
		if pivot-span.ini+1 > maxHeapLength {
			left := term{ini: span.ini, end: pivot}
			stack.Push(left)
		} else {
			PartialHeapSortFunc(sorter, span.ini, pivot, magnitude)
		}

		//right
		if span.end-pivot > maxHeapLength {
			right := term{ini: pivot + 1, end: span.end}
			stack.Push(right)
		} else {
			PartialHeapSortFunc(sorter, pivot+1, span.end, magnitude)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////        Helper fuctions from here    ///////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

func unitQuickSort(sorter Sorter, ini int, end int, magnitude func(Sorter, int) float64) int {

	startingIdxPiv, magPiv := pivoting(sorter, ini, end, magnitude)

	idxPiv := startingIdxPiv
	frontL, frontR := ini, end

	for {
		//start moving frontL from the left to find the first bigger-than- or equal-to-pivot value
		for ; frontL <= idxPiv-1; frontL++ {
			if magnitude(sorter, frontL) >= magPiv {
				break
			}
		}
		//start moving frontR from the right to find the first smaller-than-pivot value
		for ; frontR >= idxPiv+1; frontR-- {
			if magnitude(sorter, frontR) < magPiv {
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

func pivoting(sorter Sorter, ini int, end int, magnitude func(Sorter, int) float64) (idxPvt int, magPiv float64) {
	a := ini
	b := ini + (end-ini)/2
	c := end

	magA := magnitude(sorter, a)
	magB := magnitude(sorter, b)
	magC := magnitude(sorter, c)

	if magA >= magB && magA >= magC {
		if magB >= magC {
			idxPvt = b
			magPiv = magB
		} else {
			idxPvt = c
			magPiv = magC
		}
	} else if magB >= magA && magB >= magC {
		if magA >= magC {
			idxPvt = a
			magPiv = magA
		} else {
			idxPvt = c
			magPiv = magC
		}
	} else if magC >= magA && magC >= magB {
		if magA >= magB {
			idxPvt = a
			magPiv = magA
		} else {
			idxPvt = b
			magPiv = magB
		}
	}
	return
}
