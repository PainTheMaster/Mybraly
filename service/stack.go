package service

const (
	iniLength       = 16
	truncationLimit = 16
	minLength       = 16
)

//StackInterface is a interface for stack.
//The class has to implement Push(interface{}), Pop() interface{} and Depth
type StackInterface interface {
	Push(interface{})
	Pop() interface{}
	Depth() int
}

//Stack is a struct that has internal storage and works as stack
//Stack implements Push(interface{}), Pop() interface{} and Depth()
type Stack struct {
	idxTop  int //depth of the stack one Push adds 1 to this and, one pop minus 1 from this.
	length  int //length of the slice
	storage []interface{}
}

//Push adds a new element inElement to the stack.
func (s *Stack) Push(inElement interface{}) {
	if s.idxTop <= s.length-2 {
		s.idxTop++
		(s.storage)[s.idxTop] = inElement
	} else if s.idxTop == s.length-1 {
		s.storage = append(s.storage, make([]interface{}, s.length)...)
		s.length *= 2
		s.idxTop++
		(s.storage)[s.idxTop] = inElement
	} else {
		s.storage = make([]interface{}, iniLength)
		s.length = iniLength
		s.storage[s.idxTop] = inElement
	}

}

//Pop gets the top element from the top of the stack
func (s *Stack) Pop() (outElement interface{}) {
	outElement = s.storage[s.idxTop]
	s.idxTop--
	if (s.idxTop+1) <= s.length/truncationLimit && (s.idxTop+1 > minLength) {
		s.storage = s.storage[: s.idxTop+1 : s.idxTop+1]
		s.length = s.idxTop + 1
	}
	return
}

//Depth returns the depth of the stack, the number of the elements in the stack.
func (s *Stack) Depth() int {
	return s.idxTop + 1
}
