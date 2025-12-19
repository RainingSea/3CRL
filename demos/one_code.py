t = int(input().strip())

for _ in range(t):
    s = input().strip()
    # If it's already "abc", YES
    if s == "abc":
        print("YES")
        continue
    
    # Try all possible single swaps (there are only 3 choose 2 = 3 swaps)
    chars = list(s)
    possible = False
    # Swap positions (0,1), (0,2), (1,2)
    swaps = [(0,1), (0,2), (1,2)]
    for i, j in swaps:
        # Perform swap
        chars[i], chars[j] = chars[j], chars[i]
        if ''.join(chars) == "abc":
            possible = True
        # Swap back to original for next iteration
        chars[i], chars[j] = chars[j], chars[i]
    
    print("YES" if possible else "NO")