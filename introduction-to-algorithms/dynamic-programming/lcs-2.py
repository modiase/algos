def solve(seq_a: str, seq_b: str) -> str:
    m = len(seq_a)
    n = len(seq_b)
    if m < n:
        return solve(seq_b, seq_a)

    if m == 0 or n == 0:  # Handle empty sequence cases
        return ""

    # Use only two rows (current and previous)
    dp_prev = [0] * (n + 1)  # Stores values from previous row (i-1)
    dp_curr = [0] * (n + 1)  # Stores values for current row (i)

    # Forward pass: Building the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                # When characters match, we need value from previous row's diagonal (i-1, j-1)
                # This is stored in dp_prev[j-1]
                dp_curr[j] = dp_prev[j - 1] + 1
            else:
                # When characters don't match, we need max of:
                # - value from previous row, same column (i-1, j) stored in dp_prev[j]
                # - value from current row, previous column (i, j-1) stored in dp_curr[j-1]
                dp_curr[j] = max(dp_prev[j], dp_curr[j - 1])
        
        # After completing row i, current row becomes previous row for next iteration
        # We swap references instead of copying values for efficiency
        dp_prev, dp_curr = dp_curr, dp_prev

    # Backtracking phase:
    # After the final swap, dp_prev contains the values from the last computed row
    # dp_curr contains the values from the second-to-last row
    # This is why we use dp_prev[j] > dp_curr[j-1] in the backtracking comparison
    
    # Backtracking (with two rows)
    lcs = ""
    i = m
    j = n
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            lcs = seq_a[i - 1] + lcs
            i -= 1
            j -= 1
        else:
            if dp_prev[j] > dp_curr[j - 1]:  # Use dp_prev correctly
                i -= 1
            else:
                j -= 1
        # Important: Handle cases where dp_prev might not be fully updated
        if i > 0 and j > 0 and dp_prev[j] == 0 and dp_curr[j] > 0:
          i -=1
          
    return lcs



if __name__ == '__main__':
    seq_a = '001010101000010101000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
    seq_b = '10101010000010100'
    print(solve(seq_a, seq_b))