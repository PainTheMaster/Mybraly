package order

//PartialHeapSortFunc sorts the s Sorter from index ini to end by evaluating each element by using the magnitude()
//func PartialHeapSortFunc(s Sorter, ini int, end int, magnitude func(Sorter, int) float64) {
func PartialHeapSortFunc(s Sorter, ini int, end int, compare func(Sorter, int, int) int) {
	for focus := end; focus >= ini+1; focus-- {
		boss := (focus-ini-1)/2 + ini
		if compare(s, focus, boss) > 0 {
			s.Swap(focus, boss)
			for decend := focus; ini+(decend-ini)*2+1 <= end; {
				big := ini + (decend-ini)*2 + 1
				if big+1 <= end {
					if compare(s, big, big+1) < 0 {
						big = big + 1
					}
				}
				if compare(s, decend, big) < 0 {
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
				if compare(s, bigger, bigger+1) < 0 {
					bigger = bigger + 1
				}
			}
			if compare(s, focus, bigger) < 0 {
				s.Swap(focus, bigger)
				focus = bigger
			} else {
				break
			}
		}
	}
}

//PartialHeapSort sorts the s Sorter by using the Sorter.Compare()
func PartialHeapSort(s Sorter, ini int, end int) {
	defaultCompare := func(x Sorter, i int, j int) int {
		return x.Compare(i, j)
	}

	PartialHeapSortFunc(s, ini, end, defaultCompare)
}

//PartialHeapSortReverse sorts the s Sorter by using the Sorter.DefaultMagnitude()
func PartialHeapSortReverse(s Sorter, ini int, end int) {
	defalutCompare := func(x Sorter, i int, j int) int {
		return x.Compare(i, j) * (-1)
	}

	PartialHeapSortFunc(s, ini, end, defalutCompare)
}

// HeapSort sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HeapSort(s Sorter) {
	defalutCompare := func(x Sorter, i int, j int) int {
		return x.Compare(i, j)
	}

	PartialHeapSortFunc(s, 0, s.Length()-1, defalutCompare)
}

// HeapSortReverse sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HeapSortReverse(s Sorter) {
	defaultCompare := func(x Sorter, i int, j int) int {
		return x.Compare(i, j) * (-1)
	}

	PartialHeapSortFunc(s, 0, s.Length()-1, defaultCompare)
}
