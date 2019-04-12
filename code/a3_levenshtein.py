import os
import numpy as np
dataDir = '/u/cs401/A3/data/'

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                
    Examples                                                                    
    --------                                                                    
    >>> wer("who is there".split(), "is there".split())                         
    0.333 0 0 1                                                                           
    >>> wer("who is there".split(), "".split())                                 
    1.0 0 0 3                                                                           
    >>> wer("".split(), "who is there".split())                                 
    Inf 0 3 0                                                                           
    """
    n = len(r)
    m = len(h)
    R = np.zeros((n+1, m+1))
    B = np.zeros((n+1, m+1))
    for i in range(n):
        R[i][0] = i
        B[i][0] = 0
    for i in range(m):
        R[0][i] = i
        B[0][i] = 1
    total = [0,0,0]
    for i in range(n):
        for j in range(m):
            delNm = R[i-1,j]+1
            subNm = R[i-1,j-1] + (r[i-1] == h[i-1]) ? 0 : 1
            insNm = R[i, j-1] + 1
            R[i,j] = np.min(delNm, subNm, insNm)
            if R[i, j] == delNm:
                B[i, j] = "up"
            elif R[i, j] == insNm:
                B[i, j] = "left"
            else:
                B[i, j] = "up-left"
            total[0] += delNm
            total[1] += subNm
            total[2] += insNm
    return R[n, m]/n, total[1], total[2], total[0]
    # lower case and remove punc

def preproc(sent):
    sent = sent.strip().split()[2:]
    sent = " ".join(sent)
    punc = string.punctuation
    punc = punc.replace("]", "")
    punc = punc.replace("[", "")
    sent = re.sub("["+ punc +"]", "", sent)
    return sent.strip().split()

if __name__ == "__main__":
    print( 'TODO' ) 
