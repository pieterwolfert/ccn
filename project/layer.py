from activation import Tanh
from gate import AddGate, MultiplyGate
mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()


class BackPropLayer:
    """Class for implementing a backpropagation layer."""
    def forward(self, x, prev_s, U, W, V):
        """Forward pass method."""
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)
        
    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        """Backward pass method."""
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(V, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(W, prev_s, dmulw)
        dU, dx = mulGate.backward(U, x, dmulu)
        return (dprev_s, dU, dW, dV)
    
    
class FeedBackLayer:
    """Feedback Alignment layer (see Lillicrap 2014)"""
    def __init__(self, B1, B2, B3):
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        
    def forward(self, x, prev_s, U, W, V):
        """Forward pass method."""
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        """Backward pass method."""
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(self.B3, self.s, dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add, ds)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd)
        dW, dprev_s = mulGate.backward(self.B2, prev_s, dmulw)
        dU, dx = mulGate.backward(self.B1, x, dmulu)
        return (dprev_s, dU, dW, dV)



class DirectFeedBackLayer:
    """Direct feedback alignment (see Nøkland 2016)"""
    def __init__(self, B1, B2, B3):
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        
    def forward(self, x, prev_s, U, W, V):
        """Forward pass method."""
        self.mulu = mulGate.forward(U, x)
        self.mulw = mulGate.forward(W, prev_s)
        self.add = addGate.forward(self.mulw, self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V, self.s)

    def backward(self, x, prev_s, U, W, V, diff_s, dmulv):
        """Backward pass method."""
        self.forward(x, prev_s, U, W, V)
        dV, dsv = mulGate.backward(self.B3, self.s, dmulv)
        ds = dsv + diff_s
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, ds)
        dW, dprev_s = mulGate.backward(self.B2, prev_s, ds)
        dU, dx = mulGate.backward(self.B1, x, ds)
        return (dprev_s, dU, dW, dV)

