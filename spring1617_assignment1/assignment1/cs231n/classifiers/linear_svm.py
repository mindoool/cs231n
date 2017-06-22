import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dS = np.zeros([num_train, num_classes])
    for i in xrange(num_train):
        # count number of score[i, j] if negative
        negative_count = 0
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dS[i, j] = 1
            else:
                negative_count += 1
        dS[i, y[i]] = -(num_classes - negative_count - 1)
    dW = np.transpose(X).dot(dS)
    dW /= num_train

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * W * reg

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_classes = W.shape[1]
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    dS = np.zeros([num_train, num_classes])
    scores = X.dot(W)
    # scores_correct = (N,)
    scores_correct = scores[np.arange(num_train), y]
    # margins = (scores.T - scores_correct).T + 1
    margins = scores - scores_correct.reshape((num_train,1)) + 1
    # set 0 if j = y[i]
    margins[np.arange(num_train), y] = 0
    # set 0 if margin is negative
    forward_margins = np.maximum(0, margins)
    loss = np.sum(forward_margins) / num_train
    loss += reg * np.sum(W * W)
    
    # 0보다 작은 margin갯수를 row별로 더하기
    dS[margins < 0] = 1
    negative_count = np.sum(dS, axis=1)
    # margin값이 0보다 크면 1, 0보다 작으면 0 (default)
    dS[margins > 0] = 1
    dS[margins < 0] = 0
    # j=y[i]이면 -(C-negative_count-1)
    dS[np.arange(num_train), y] = -(-negative_count - 1 + W.shape[1])
    # dW
    dW = np.transpose(X).dot(dS)
    dW /= num_train
    dW += 2 * W * reg

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
