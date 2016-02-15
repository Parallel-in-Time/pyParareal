from timemesh import timemesh

class parareal(object):

    def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max):
      self.timemesh = timemesh(tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max)

    def run(self, u0):

      # Coarse predictor
      self.timemesh.run_coarse(u0)

      while not self.timemesh.all_converged():

    #
    # GET functions
    #
    def get_parareal_matrix(self):

    # Return fine value of last time slices
    def get_final_value(self):
      return self.timemesh.get_fine_value(self.timemesh.nslices-1)
