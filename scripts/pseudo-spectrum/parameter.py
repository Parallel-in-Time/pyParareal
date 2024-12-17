class parameter(object):
    
    def __init__(self, dedalus):
        self.Tend    = 1.0
        self.nslices = 10
        self.tol = 0.0        
        self.maxiter = 9
        self.nfine   = 10
        if dedalus:
            self.ncoarse =  10
        else:
            self.ncoarse = 1
        self.epsilon = 0.1
        self.ndof_f   = 32
        
    def getpar(self):
        return self.Tend, self.nslices, self.maxiter, self.nfine, self.ncoarse, self.tol, self.epsilon, self.ndof_f
        
