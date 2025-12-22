from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        INF = 10**18
        
        # dp[a][b] = minimum length with starting char a and ending char b
        dp = [[INF] * 26 for _ in range(26)]
        
        first_word = words[0]
        f = ord(first_word[0]) - ord('a')
        l = ord(first_word[-1]) - ord('a')
        dp[f][l] = len(first_word)
        
        for w in words[1:]:
            nf = ord(w[0]) - ord('a')
            nl = ord(w[-1]) - ord('a')
            wlen = len(w)
            
            new_dp = [[INF] * 26 for _ in range(26)]
            
            for a in range(26):
                for b in range(26):
                    if dp[a][b] == INF:
                        continue
                    
                    cur = dp[a][b]
                    
                    # join(current, w)
                    cost1 = cur + wlen - (1 if b == nf else 0)
                    new_dp[a][nl] = min(new_dp[a][nl], cost1)
                    
                    # join(w, current)
                    cost2 = cur + wlen - (1 if nl == a else 0)
                    new_dp[nf][b] = min(new_dp[nf][b], cost2)
            
            dp = new_dp
        
        return min(min(row) for row in dp)

if __name__=="__main__":
    a = Solution()
    print(a.minimizeConcatenatedLength(["aa", "ab", "bc"]))
    # print(a.minimizeConcatenatedLength([["aa", "ab", 'bc']]))
    