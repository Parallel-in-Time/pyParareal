from timeslice import timeslice
from special_integrator import special_integrator
import numpy as np
from scipy.sparse import linalg
from scipy import sparse
from solution import solution
import copy

class timemesh(object):

  def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max, u0fine, u0coarse):
    assert tstart<tend, "tstart has to be smaller than tend"
    #
    # For the time being, all timeslices are created equal...
    #
    self.timemesh = np.linspace(tstart, tend, nslices+1)
    self.nslices  = nslices
    self.slices   = []
    self.tstart   = tstart
    self.tend     = tend    

    # @NOTE: this setup would allow to set different values for tolerance and iter_max for different slices...
    # ... however, this has not yet been tested!!
    for i in range(0,nslices):
     
      if sparse.issparse(fine):
        ts_fine = special_integrator(self.timemesh[i], self.timemesh[i+1], nsteps_fine, fine)
      else:
        ts_fine   =   fine(self.timemesh[i], self.timemesh[i+1], nsteps_fine)
      
      if sparse.issparse(coarse):
        ts_coarse = special_integrator(self.timemesh[i], self.timemesh[i+1], nsteps_coarse, coarse)      
      else:
        ts_coarse = coarse(self.timemesh[i], self.timemesh[i+1], nsteps_coarse)
      
      self.slices.append( timeslice(ts_fine, ts_coarse, tolerance, iter_max, u0fine, u0coarse) )

  # Run the coarse method serially over all slices
  def run_coarse(self, u0):
    self.set_initial_value(u0)
    for i in range(0,self.nslices):
      # Run coarse method
      self.slices[i].update_coarse()
      # Fetch coarse value and set initial value of next slice
      if i<self.nslices-1:
        self.set_initial_value( copy.deepcopy(self.get_coarse_value(i)), i+1 )

  # Run the fine method serially over all slices
  def run_fine(self, u0):
    self.set_initial_value(u0)
    for i in range(0,self.nslices):
      # Run fine method
      self.slices[i].update_fine()
      # Fetch fine value and set initial value of next slice
      if i<self.nslices-1:
        self.set_initial_value( copy.deepcopy(self.get_fine_value(i)), i+1 )

  # Update fine values for all slices
  # @NOTE: This is not equivalent to run_fine, since no updated initial values are copied forward
  def update_fine_all(self):
    for i in range(0,self.nslices):
      self.slices[i].update_fine()

  # Update coarse values for all slices
  # @NOTE: This is not equivalent to run_fine, since no updated initial values are copied forward
  def update_coarse_all(self):
    for i in range(0,self.nslices):
      self.slices[i].update_coarse()

  '''
  Updates the coarse value for one time slice
  '''
  def update_coarse(self, i):
    self.slices[i].update_coarse()

  '''
  Updates the fine value for one time slice
  '''
  def update_fine(self, i):
    self.slices[i].update_fine()

  #
  # SET functions
  #
  def set_initial_value(self, u0, slice_nr=0):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    self.slices[slice_nr].set_sol_start(u0)

  def set_end_value(self, u0, slice_nr=0):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    self.slices[slice_nr].set_sol_end(u0)

  # increase iteration counter of a single time slice
  def increase_iter(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    self.slices[slice_nr].increase_iter()

  # increase iteration counters of ALL time slices
  def increase_iter_all(self):
    for i in range(0,self.nslices):
      self.increase_iter(i)

  #
  # GET functions
  #
  def get_coarse_value(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    return self.slices[slice_nr].get_sol_coarse()

  def get_fine_value(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    return self.slices[slice_nr].get_sol_fine()

  def get_end_value(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    return self.slices[slice_nr].get_sol_end()

  def get_max_residual(self):
    maxres = self.slices[0].get_residual()
    for i in range(1,self.nslices-1):
      maxres = max( maxres, self.slices[i].get_residual() )
    return maxres

  # Returns a matrix such that inversion by block elimination corresponds to 
  # running the fine method in serial:
  # | Id              | | u_0 |   | u_0 |
  # | -F Id           | | u_1 |   |   0 |
  # |    -F Id        | | u_2 | = | ... | 
  # |       ... ...   | | ... |   |     |
  # |            -F Id| | u_N |   |   0 |
  def get_fine_matrix(self, u0):
    assert isinstance(u0, solution), "Argument u0 must be an object of type solution."
    Id = sparse.eye( u0.ndof, format="csc" )
    Fmat = Id
    for i in range(0,self.nslices):
      if i==0:
        lower = -self.slices[i].get_fine_update_matrix(u0)
      else:
        lower = sparse.csc_matrix(0.0*lower)
        lower = sparse.hstack([lower, -self.slices[i].get_fine_update_matrix(u0)])
      Fmat = sparse.bmat([[Fmat, None],[lower, Id]], format="csc")
    return Fmat

  # Returns a matrix such that inversion by block elimination corresponds to running the coarse method in serial
  def get_coarse_matrix(self, u0):
    assert isinstance(u0, solution), "Argument u0 must be an object of type solution."
    Id = sparse.eye( self.slices[0].ndof_f, format="csc" ) # PROBLEM
    Cmat = Id
    for i in range(0,self.nslices):
      if i==0:
        lower = -self.slices[i].get_coarse_update_matrix(u0)
      else:
        lower = sparse.csc_matrix(0.0*lower)
        lower = sparse.hstack([lower, -self.slices[i].get_coarse_update_matrix(u0)])
      Cmat = sparse.bmat([[Cmat, None],[lower, Id]], format="csc")
    return Cmat

  #
  # IS functions
  #   
  def all_converged(self):
    all_converged = True
    i = 0
    while (all_converged and i<self.nslices):
      if not self.slices[i].is_converged():
        all_converged = False
      i += 1
    return all_converged
