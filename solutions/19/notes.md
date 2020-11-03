# Notes

## Potential Algorithm

1. Select cheapest colour in each row to create state k<sub>0</sub>.
2. Handle adjacent conflict by changing the colour of the house for which it is the cheapest to do so to create k<sub>1</sub>.
3. If this creates a new conflict then compare the result of performing step 2 on k<sub>1</sub> (to produce k<sub>2</sub>) with the result of k<sub>1'=</sub> which is created by changing the colour of the other conflicting house.
4. If k<sub>1'=</sub> also produces a conflict then perform 2 on k<sub>1'=</sub> to produce k<sub>2'=</sub>.
5. If k<sub>2'=</sub> is a conflict then we check k<sub>1'='=</sub> with k<sub>3'=</sub>.
6. Repeat the above steps of normal and prime moves until a solution is found in each branch.
7. Find the lowest priced solution at each branch node.
