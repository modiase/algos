
class Node:
    def __init__(self, payload, l = None, r = None):
        self.payload = payload
        self.l = l
        self.r = r
    def __str__(self):
        return(f'<Node({self.payload} L<{self.l}> R<{self.r}>')

def read_mat_from_file(in_fp=None):
    if not in_fp:
        raise RuntimeError("No supplied filepath.")
    m = []
    with open(in_fp,'r') as f:
        for line in f.readlines():
            m.append([ float(n) for n in line.replace('\n','').split(' ')])
    return m