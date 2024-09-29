import numpy as np

def isUnique(seq1, seq2, cutoff = 1):
    unique = False
    num_SNPs = 0
    idx = []
    if len(seq1) != len(seq2):
        raise NameError('contigs must be aligned')
    else:
        tot_1 = 0
        tot_2 = 0
        for i in range(len(seq1)):
            if seq1[i] == 'A' or seq1[i] == 'C' or seq1[i] == 'T' or seq1[i] == 'G':
                tot_1 = tot_1 + 1
            if seq2[i] == 'A' or seq2[i] == 'C' or seq2[i] == 'T' or seq2[i] == 'G':
                tot_2 = tot_2 + 1
            if seq1[i] != seq2[i]: # if they aren't equal to each other
                if (seq1[i] == 'A' or seq1[i] == 'C' or seq1[i] == 'T' or seq1[i] == 'G') and (seq2[i] == 'A' or seq2[i] == 'C' or seq2[i] == 'T' or seq2[i] == 'G'): #and are also A, C, T, G (ignore other letters for now)
                    #print('at position ' + str(i) + ', seq1 is ' + str(seq1[i]) + ' while seq2 is ' + str(seq2[i]))
                    unique = True
                    num_SNPs = num_SNPs + 1
                    idx.append(i)
        if num_SNPs <= cutoff:
            unique = False
        percent_diff = num_SNPs/np.min([tot_1, tot_2])
        return unique, num_SNPs, idx, percent_diff
