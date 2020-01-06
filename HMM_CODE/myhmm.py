import numpy as np
import logging
import math
import pdb
from sklearn.metrics import f1_score

FORMAT = "[%(asctime)s] : %(filename)s.%(funcName)s():%(lineno)d - %(message)s"
DATEFMT = '%H:%M:%S, %m/%d/%Y'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, datefmt=DATEFMT)
logger = logging.getLogger(__name__)
np.set_printoptions(linewidth=200, precision=3, suppress=True)
debug = False

def learn_HMM(pi, A, B, iterlimit=1000, threshold=0.0001):
    """
    This estimates the parameters lambda of an HMM using only an emitted
    symbol sequence O (unsupervised) and an initial guess lambda_0.
    Parameters:
      pi - a 1XN dimensional numpy array, the initial probabilities array,
           initialized randomly, sums to 1
      A - an NxN dimensional numpy array, the transition matrix,
          initialized randomly, each row sums to 1
      B - an NXM dimensional numpy array, the state-observation matrix,
          initialized randomly, each row sums to 1
      O - a 1XT dimensional numpy array, the observations,
          This must consist of integers chosen from a discrete set. In
          particular, if there are 5 possible observations, then the
          set must be {0, 1, 2, 3, 4}.
    """

    converged = False
    T = B.shape[1] # time
    N = len(A) # states
    M = len(B[0]) # features / len of O
    cnt = 0
    logPold = -np.infty
    iters = 0
    P_list = []
    logP_list = []
    while not converged and iters < iterlimit:
        alpha = np.zeros((T, N)) # 17299, 6
        beta = np.zeros((T, N))
        gamma = np.zeros((T, N))
        xi = np.zeros((N, N, T - 1))
        c = np.zeros(T) # scaling at each t
        iters += 1

        # Compute alpha (scaled)
        for i in range(N):
            alpha[0, i] = pi[i] * B[i, 0]
        c[0] = 1. / np.sum(alpha[0, :])
        alpha[0, :] *= c[0]
        for t in range(1, T):
            for j in range(N):
                sm = 0
                for i in range(N):
                    sm += alpha[t - 1][i] * A[i, j]
                alpha[t, j] = sm * B[j, t]
            c[t] = 1. / np.sum(alpha[t, :])
            alpha[t, :] *= c[t]

        # Compute beta (scaled)
        for j in range(N):
            beta[T - 1, j] = 1.
        beta[T - 1, :] *= c[T - 1]
        for t in range(T - 2, -1, -1):
            for i in range(N):
                sm = 0
                for j in range(N):
                    sm += A[i, j] * B[j, t+1] * beta[t + 1, j]
                beta[t, i] = sm
            beta[t, :] *= c[t]

        if debug:
            print "alpha: (first few lines)"
            print alpha[:5, :]
            print "\nbeta: (first few lines)"
            print beta[:5, :]

        # compute gammas
        for t in range(T):
            rowsum = np.dot(alpha[t, :], beta[t, :]) # sum of states at t - P
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i] / rowsum

        # compute xi
        for t in range(T - 1):
            xisum = 0
            for i in range(N):
                for j in range(N):
                    xi[i, j, t] = alpha[t, i] * A[i, j] * B[j, t+1] * beta[t + 1, j]
                    xisum += xi[i, j, t] # - P
            xi[:, :, t] /= xisum

        # Following after description in Levinson book
        # Update A
        for i in range(N):
            for j in range(N):
                num = 0
                denom = 0
                for t in range(T - 1):
                    num += xi[i, j, t]
                    denom += gamma[t, i]
                A[i, j] = num / denom

        # update B
        #for j in range(N):
        #    denom = np.sum(gamma[:, j])
        #    for k in range(M):
        #        num = 0
        #        for t in range(T):
        #            if O[t] == k:
        #                num += gamma[t, j]
        #        B[j, k] = num / denom

        # update pi
        pi = gamma[0, :]

        print "Iteration " + str(cnt + 1)

        logP = -1 * np.sum(map(lambda ct: math.log(ct), c))
        print "logP: ", logP
        logP_list.append(logP)
        diff = logP - logPold
        print "change in prob (should be positive): ", diff, "\n"
        if diff < 0:
            "ERROR: diff is not positive!"
            break
        if diff < threshold:
            print "We have reached our goal. diff=", diff
            converged = True
        logPold = logP

        if debug:
            print "pi:"
            print pi, sum(pi)
            print "A:"
            for a in A:
                print a, sum(a)
            print "B:"
            for b in B:
                print b, sum(b)
        cnt += 1
    return pi, A, B, logP_list

def test_HMM_viterbi(M, N, newPi, newA, newB, y, lamb):
    phi = np.zeros((M, N))
    state_sequence = -np.ones((M, N))
    for i in range(N):
        phi[0, i] = np.log(newPi[i] * newB[i, 0])
    for t in range(1, M):
        for j in range(N):
            max_value = -np.infty
            max_state = -1
            for i in range(N):
                if phi[t - 1][i] + np.log(newA[i, j]) > max_value:
                    max_value = phi[t - 1][i] + np.log(newA[i, j])
                    max_state = i
            phi[t, j] = max_value + lamb * np.log(newB[j, t])
            state_sequence[t, j] = max_state # best state to [t,j]

    viterbi_logP = np.max(phi[M-1, :])
    viterbi_state = np.argmax(phi[M-1, :])
    best_sequence = [viterbi_state]
    for t in range(M-1, 0, -1):
        viterbi_state = int(state_sequence[t, viterbi_state])
        best_sequence.append(viterbi_state)
    best_sequence.reverse()
    best_sequence = np.array(best_sequence)
    y = np.array(y)
    updated_accuracy = sum(best_sequence == y) * 1.0/len(y)
    FSCORE = f1_score(y, best_sequence, average='macro')
    return updated_accuracy, best_sequence, FSCORE
