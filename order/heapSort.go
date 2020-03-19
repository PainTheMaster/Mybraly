package order

//PartialHeapSortFunc sorts the s Sorter from index ini to end by evaluating each element by using the magnitude()
func PartialHeapSortFunc(s Sorter, ini int, end int, magnitude func(Sorter, int) float64) {

	for focus := end; focus >= ini+1; focus-- {
		boss := (focus-ini-1)/2 + ini
		if magnitude(s, focus) > magnitude(s, boss) {
			s.Swap(focus, boss)
			for decend := focus; ini+(decend-ini)*2+1 <= end; {
				big := ini + (decend-ini)*2 + 1
				if big+1 <= end {
					if magnitude(s, big) < magnitude(s, big+1) {
						big = big + 1
					}
				}
				if magnitude(s, decend) < magnitude(s, big) {
					s.Swap(decend, big)
					decend = big
				} else {
					break
				}
			}
		}
	}

	for fix := end; fix >= ini+1; fix-- {
		s.Swap(fix, ini)
		for focus := ini; ini+(focus-ini)*2+1 <= fix-1; {
			bigger := ini + (focus-ini)*2 + 1
			if bigger+1 <= fix-1 {
				if magnitude(s, bigger) < magnitude(s, bigger+1) {
					bigger = bigger + 1
				}
			}
			if magnitude(s, focus) < magnitude(s, bigger) {
				s.Swap(focus, bigger)
				focus = bigger
			} else {
				break
			}
		}
	}
}

//PartialHeapSort sorts the s Sorter by using the Sorter.DefaultMagnitude()
func PartialHeapSort(s Sorter, ini int, end int) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i)
	}

	PartialHeapSortFunc(s, 0, s.Length(), dummyMagnitude)
}

//PartialHeapSortReverse sorts the s Sorter by using the Sorter.DefaultMagnitude()
func PartialHeapSortReverse(s Sorter, ini int, end int) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i) * (-1.0)
	}

	PartialHeapSortFunc(s, 0, s.Length(), dummyMagnitude)
}

// HeapSort sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HeapSort(s Sorter) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i)
	}

	PartialHeapSortFunc(s, 0, s.Length()-1, dummyMagnitude)
}

// HeapSortReverse sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HeapSortReverse(s Sorter) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i) * (-1.0)
	}

	PartialHeapSortFunc(s, 0, s.Length()-1, dummyMagnitude)
}
