import numpy as np
from rnn_utils import *


def custom_rnn1_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: it/ot stand for the update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wi = parameters["Wi"] # update gate weight 
    bi = parameters["bi"] # 
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for it, cct, c_next, ot, a_next using the formulas given 
    it = sigmoid(np.dot(Wi,concat) + bi) # update gate
    cct = np.tanh(np.dot(Wc, concat) + bc) # candidate value 
    c_next = np.multiply((1-it), c_prev) + np.multiply(it, cct) # new cell state
    ot = sigmoid(np.dot(Wo,concat) + bo) # output gate 
    a_next = np.multiply(ot, np.tanh(c_next)) # new hidden state
    
    # Compute prediction of the LSTM cell 
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def custom_rnn1_forward(x, a0, parameters):
    """
    Implement the forward propagation of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    # a_next, c_next, yt_pred, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    # Initialize "caches", which will track the list of all the caches
    caches = []

    Wy = parameters['Wy'] 
    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache 
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state
        c[:,:,t]  = c_next
        # Save the value of the prediction in y 
        y[:,:,t] = yt
        # Append the cache into caches 
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches




def custom_rnn2_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        __
                        Wf2 -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf2 -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        --
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"] # forget gate weight
    bf = parameters["bf"]
    
    Wf2 = parameters["Wf2"] # forget gate weight
    bf2 = parameters["bf2"]
    
    Wi = parameters["Wi"] # update gate weight 
    bi = parameters["bi"] # 
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for ft, it, cct, c_next, ot, a_next 
    ft = sigmoid(np.dot(Wf,concat) + bf) # forget gate
    ft2 = sigmoid(np.dot(Wf2,concat) + bf2) # forget gate 2
    it = sigmoid(np.dot(Wi,concat) + bi) # update gate
    cct = np.tanh(np.dot(Wc, concat) + bc) # candidate value 
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct) # new cell state
    ot = sigmoid(np.dot(Wo,concat) + bo) # output gate 
    a_next = np.multiply(ot, np.tanh(c_next + np.multiply(ft2, c_prev))) # new hidden state
    
    # Compute prediction of the LSTM cell 
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, ft2, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache



def custom_rnn2_forward(x, a0, parameters):
    """
    Implement the forward propagation of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        __
                        Wf2 -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf2 -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        --
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo -- Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    # a_next, c_next, yt_pred, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    # Initialize "caches", which will track the list of all the caches
    caches = []

    Wy = parameters['Wy'] 
    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache 
        a_next, c_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state
        c[:,:,t]  = c_next
        # Save the value of the prediction in y 
        y[:,:,t] = yt
        # Append the cache into caches 
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches




def custom_rnn3_cell_forward(xt, a_prev, c_prev, parameters):
    """
    Implement a single forward step of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
    c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        
                        Wi2 -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi2 -- Bias of the update gate, numpy array of shape (n_a, 1)
                        
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        
                        Wc2 -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc2 --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        
                        
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        
                        Wo2 -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo2 --  Bias of the output gate, numpy array of shape (n_a, 1)
                        
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
                        
    Returns:
    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
    
    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    # Retrieve parameters from "parameters"
    Wi = parameters["Wi"] # update gate weight 
    bi = parameters["bi"] #
    
    Wi2 = parameters["Wi2"] # update gate weight 
    bi2 = parameters["bi2"] #
    
    Wc = parameters["Wc"] # candidate value weight
    bc = parameters["bc"]
    
    Wc2 = parameters["Wc2"] # candidate value weight
    bc2 = parameters["bc2"]
    
    Wo = parameters["Wo"] # output gate weight
    bo = parameters["bo"]
    
    Wo = parameters["Wo2"] # output gate weight
    bo = parameters["bo2"]
    
    
    Wy = parameters["Wy"] # prediction weight
    by = parameters["by"]
    
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈1 line)
    concat = np.concatenate((a_prev, xt), axis=0)

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) 
    it = sigmoid(np.dot(Wi,concat) + bi) # update gate
    it2 = sigmoid(np.dot(Wi2,concat) + bi2) # update gate
    
    cct = np.tanh(np.dot(Wc, concat) + bc) # candidate value 
    c_next = np.multiply(it, c_prev) + np.multiply((1-it), cct) # new cell state
    
    c2ct = np.tanh(np.dot(Wc2, concat) + bc2) # candidate value 
    c2_next = np.multiply(it2, c_prev) + np.multiply((1-it2), c2ct) # new cell state
    
    ot = sigmoid(np.dot(Wo,concat) + bo) # output gate 
    ot2 = sigmoid(np.dot(Wo2,concat) + bo2) # output gate 
    
    a_next = np.multiply(ot, np.tanh(c_next)) +  np.multiply(ot2, np.tanh(c2_next)) # new hidden state
    
    # Compute prediction of the LSTM cell 
    yt_pred = softmax(np.dot(Wy, a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, it, it2, cct, c2ct, ot, ot2, xt, parameters)

    return a_next, c_next, c2_next, yt_pred, cache



def custom_rnn3_forward(x, a0, parameters):
    """
    Implement the forward propagation of my augmented custom RNN cell as described by rnn_equations.ipynb

    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    a0 -- Initial hidden state, of shape (n_a, m)
    parameters -- python dictionary containing:
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        
                        Wi2 -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi2 -- Bias of the update gate, numpy array of shape (n_a, 1)
                        
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        
                        Wc2 -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc2 --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        
                        
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        
                        Wo2 -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo2 --  Bias of the output gate, numpy array of shape (n_a, 1)
                        
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    # a_next, c_next, yt_pred, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    # Initialize "caches", which will track the list of all the caches
    caches = []

    Wy = parameters['Wy'] 
    # Retrieve dimensions from shapes of x and parameters['Wy']
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape
    
    # initialize "a", "c" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    c2 = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    
    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))
    
    # loop over all time-steps
    for t in range(T_x):
        # Get the 2D slice 'xt' from the 3D input 'x' at time step 't'
        xt = x[:,:,t]
        # Update next hidden state, next memory state, compute the prediction, get the cache 
        a_next, c_next, c2_next, yt, cache = lstm_cell_forward(xt, a_next, c_next, c2_next, parameters)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:,:,t] = a_next
        # Save the value of the next cell state
        c[:,:,t]  = c_next
        c2[:,:,t]  = c2_next
        # Save the value of the prediction in y 
        y[:,:,t] = yt
        # Append the cache into caches 
        caches.append(cache)
    
    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches