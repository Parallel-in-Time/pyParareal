from integrator import integrator
from solution import solution
from solution_linear import solution_linear

class special_integrator(integrator):

  def __init__(self, tstart, tend, nsteps, stab_function):
    super(special_integrator, self).__init__(tstart, tend, nsteps)
    self.stab_function = stab_function

  def run(self, u0):
    assert isinstance(u0, solution_linear), "special_integrator can only be run for solutions of type solution_linear"
    for i in range(0,self.nsteps):
      u0.y = self.stab_function@u0.y

  def get_update_matrix(self, sol):
    return self.stab_function**self.nsteps
