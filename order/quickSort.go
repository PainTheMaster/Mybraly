package order

import "PainTheMaster/mybraly/service"

type term struct {
	ini int
	end int
}

// QuickSort sorts sorter Sorter by quick sort method
func QuickSort(sorter Sorter, ini int, end int, magnitude func(Sorter, int) float64) {
	var stack service.Stack
	var iniEnd term

	iniEnd.ini = ini
	iniEnd.end = end
	stack.Push(iniEnd)

}

func unitQuichSort(sorter Sorter, ini int, end int, magnitude func(Sorter, int) float64) (idxPvt int) {
	return
	startingIdxPiv, magPiv := pivoting(sorter, ini, end, magnitude)

	idxPiv := startingIdxPiv
	frontL, frontR := ini-1, end+1

	return
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
