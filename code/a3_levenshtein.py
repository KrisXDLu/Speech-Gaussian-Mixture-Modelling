import os
import numpy as np
import string
import re
from scipy import stats

# dataDir = '/u/cs401/A3/data/'
dataDir = "../data/"

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
    for i in range(1, n+1):
        R[i][0] = i
        B[i][0] = 0
    for i in range(1, m+1):
        R[0][i] = i
        B[0][i] = 1
    total = [0,0,0]
    for i in range(1, n+1):
        for j in range(1, m+1):
            delNm = R[i-1,j]+1
            if r[i-1] == h[j-1]:
                subNm = R[i-1,j-1] + 0 
            else:
                subNm = R[i-1,j-1] + 1
            insNm = R[i, j-1] + 1
            R[i,j] = min(delNm, subNm, insNm)
            if R[i, j] == delNm:
                B[i, j] = 0
            elif R[i, j] == insNm:
                B[i, j] = 1
            else:
                B[i, j] = 3 if r[i-1] == h[j-1] else 2
    # backtrack
    while j > 0 or i > 0:
        if B[i, j] == 0:
            total[2] += 1
            i -= 1
        elif B[i, j] == 1:
            total[1] += 1
            j -= 1
        elif B[i, j] == 2:
            total[0] += 1
            i -= 1
            j -= 1
        else:
            i -= 1
            j-= 1
    return R[n, m]/n, total[0], total[1], total[2]
    # lower case and remove punc

def preproc(sent):
    sent = sent.strip().split()[2:]
    sent = " ".join(sent)
    punc = string.punctuation
    punc = punc.replace("]", "")
    punc = punc.replace("[", "")
    sent = re.sub("["+ punc +"]", "", sent)
    sent = sent.lower()
    return sent.strip().split()



if __name__ == "__main__":
    f = open('asrDiscussion.txt', 'w')

    google = []
    kaldi = []
    for path, dirnames, filenames in os.walk(dataDir):
        for speaker in dirnames:
            path = os.path.join(dataDir, speaker)
            googF = open(path + '/transcripts.Google.txt', 'r').read().splitlines()
            kaldiF = open(path + '/transcripts.Kaldi.txt', 'r').read().splitlines()
            referF = open(path + '/transcripts.txt', 'r').read().splitlines()

            iterNm = min(len(referF), len(kaldiF), len(googF))
            for i in range(iterNm):
                kaldiSent = preproc(kaldiF[i])
                googSent = preproc(googF[i])
                referSent = preproc(referF[i])
                # print(googSent)

                gWER, gNS, gNI, gND = Levenshtein(referSent, googSent)
                kWER, kNS, kNI, kND = Levenshtein(referSent, kaldiSent)
                google.append(gWER)
                kaldi.append(kWER)

                f.write('{} {} {} {} S:{}, I:{}, D:{}\n'.format(speaker, 'Google', i, gWER, gNS, gNI, gND))
                f.write('{} {} {} {} S:{}, I:{}, D:{}\n'.format(speaker, 'Kaldi', i, kWER, kNS, kNI, kND))

    f.write('Google WER mean: {} Google WER standard deviation: {}\n').format(np.mean(google), np.std(google))
    f.write('Kaldi WER mean: {} Kaldi WER standard deviation: {}\n'.format(np.mean(kaldi), np.std(kaldi)))
    # f.write()
    f.close()
