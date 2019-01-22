import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        # Subtract max from scores to ensure numeric stability
        scores -= np.max(scores)
        # Preform softmax operations
        scores = np.exp(scores) / np.exp(scores).sum()
        loss += -np.log(scores[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (scores[j] - 1) * X[i]
            else:
                dW[:, j] += scores[j] * X[i]

    # Averaging over training set
    loss /= num_train
    dW /= num_train

    # Adding regularization terms
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    
    scores = X.dot(W)
    # Subtract max from scores to ensure numeric stability
    scores -= np.max(scores, axis=1, keepdims=True)
    # Preform softmax operations
    scores = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    loss += np.sum(-np.log(np.choose(y, scores.T)))
    
    # Gradient consists of two parts: X * scores for each weight and addidional
    # -X term for classes y[i]
    # Here generate mask of ones for the second term to subtract from scores
    mask = np.zeros_like(scores)
    mask[np.arange(scores.shape[0]), y] = 1 # Has 1 on the (i, y[i]) positions
    dW += X.T.dot(scores - mask)

    # Averaging over training set
    loss /= num_train
    dW /= num_train

    # Adding regularization terms
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

