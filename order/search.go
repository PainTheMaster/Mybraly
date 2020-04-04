package order

//LinearSearch perfoms linear search
//The srt has to be sorted prior to appling this function
func LinearSearch(srt Sorter, customCompare func(Sorter, int) int) (idxFound int) {
	idxFound = -1

	for idx := 0; idx <= srt.Length()-1; idx++ {
		if customCompare(srt, idx) == 0 {
			idxFound = idx
			break
		}
	}
	return
}

//BinarySearch performs BinarySearch
//srt has to be sorted prior to search.
//func shoud return 1 if the member designated by the argument int is smaller than a standard
func BinarySearch(srt Sorter, customCompare func(Sorter, int) int) (idxFound int) {
	const minBinaryLength = 16

	idxFound = -1

	left := 0
	right := srt.Length() - 1
	middle := left + (right-left)/2

	for {
		if right-(left-1) >= minBinaryLength {
			if customCompare(srt, middle) < 0 {
				right = middle
				middle = left + (right-left)/2
			} else if customCompare(srt, middle) > 0 {
				left = middle
				middle = left + (right-left)/2
			} else {
				idxFound = middle
				break
			}
		} else {
			idxFound = LinearSearch(srt, customCompare)
			break
		}
	}
	return
}
