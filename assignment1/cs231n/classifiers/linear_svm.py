import numpy as np
from random import shuffle

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # Count how many times we need to update dW[:, y[i]]
        grad_count = 0
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                continue
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                grad_count += 1
        
        dW[:, y[i]] += - grad_count * X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Same for gradient
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization to the gradient
    dW += 2 * reg * W
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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################    
    num_train = X.shape[0]
    scores = X.dot(W)
    # Selecting scores of the right answers
    y_i_scores = np.choose(y, scores.T)
    # At this point all scores are correct except for the scores of y_i
    margins = scores - y_i_scores.reshape(-1, 1) + 1
    # Here we generate mask for y_i scores, combine it 
    # with mask for positive margins and apply it
    mask = np.zeros_like(scores, dtype=np.bool)
    mask[np.arange(num_train), y] = 1
    full_mask = ~mask * (margins>0)
    margins *= full_mask
    
    loss = margins.sum() / num_train
    loss += reg * np.sum(W * W)
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
    # The gradient values are is X[i] for all gradients except y_i (X * full_mask)
    # and X[i] * (number of positive margins) for y_i, which is
    # (X * (mask * full_mask.sum(axis=1)))
    grad_multiplier = full_mask - mask * full_mask.sum(axis=1, keepdims=True)
    dW = X.T.dot(grad_multiplier) / num_train
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
