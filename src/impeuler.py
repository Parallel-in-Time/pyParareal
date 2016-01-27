from integrator import integrator

class impeuler(integrator):

  def __init__(self, tstart, tend, nsteps):
    super(impeuler, self).__init__(tstart, tend, nsteps)
