package mysort

//Sorter is a interface whose element's magnitude can be evaluated and be swapped with another element
type Sorter interface {
	DefaultMagnitude(int) float64
	Swap(int, int)
	Length() int
}

//PartialHashSortFunc sorts the s Sorter from index ini to end by evaluating each element by using the magnitude()
func PartialHashSortFunc(s Sorter, ini int, end int, magnitude func(Sorter, int) float64) {

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

//PartialHashSort sorts the s Sorter by using the Sorter.DefaultMagnitude()
func PartialHashSort(s Sorter, ini int, end int) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i)
	}

	PartialHashSortFunc(s, 0, s.Length(), dummyMagnitude)
}

//PartialHashSortReverse sorts the s Sorter by using the Sorter.DefaultMagnitude()
func PartialHashSortReverse(s Sorter, ini int, end int) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i) * (-1.0)
	}

	PartialHashSortFunc(s, 0, s.Length(), dummyMagnitude)
}

// HashSort sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HashSort(s Sorter) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i)
	}

	PartialHashSortFunc(s, 0, s.Length()-1, dummyMagnitude)
}

// HashSortReverse sorts s Sorter from index 0 to the end by using the Sorter.DefaltMagnitude()
func HashSortReverse(s Sorter) {
	dummyMagnitude := func(x Sorter, i int) float64 {
		return x.DefaultMagnitude(i) * (-1.0)
	}

	PartialHashSortFunc(s, 0, s.Length()-1, dummyMagnitude)
}
