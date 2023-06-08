from solution import is_valid, mat_tot, init_solution

init = [[1,1,1],[0,0,0],[0,0,0]]
mat = [[1,2,3],[4,5,6],[7,8,9]]

def test_is_valid():
    assert not is_valid([[1,0,0],[1,0,0],[0,0,1]])
    assert is_valid([[0,1,0],[1,0,0],[0,1,0]])
    assert is_valid([[0,1,0],[1,0,0],[0,0,1]])
    assert not is_valid([[0,1,0],[0,1,0],[0,1,0]])

def test_mat_tot():
    assert mat_tot(init,mat) == 12

def test_init():
    assert init_solution(mat) == init

def test_generate_tree():
    ...

def test_move():
    ...