from timeslice import timeslice
import numpy as np
from scipy.sparse import linalg
from scipy import sparse

class timemesh(object):

  def __init__(self, tstart, tend, nslices, fine, coarse, nsteps_fine, nsteps_coarse, tolerance, iter_max):
    assert tstart<tend, "tstart has to be smaller than tend"
    #
    # For the time being, all timeslices are created equal...
    #
    self.timemesh = np.linspace(tstart, tend, nslices+1)
    self.nslices  = nslices
    self.slices   = []

    for i in range(0,nslices):
      ts_fine   =   fine(self.timemesh[i], self.timemesh[i+1], nsteps_fine)
      ts_coarse = coarse(self.timemesh[i], self.timemesh[i+1], nsteps_coarse)
      self.slices.append( timeslice(ts_fine, ts_coarse, tolerance, iter_max) )

  # Run the coarse method serially over all slices
  def run_coarse(self, u0):
    self.set_initial_value(u0)
    for i in range(0,self.nslices):
      # Run coarse method
      self.slices[i].update_coarse()
      # Fetch coarse value and set initial value of next slice
      if i<self.nslices-1:
        self.set_initial_value( self.get_coarse_value(i), i+1 )

  # Run the fine method serially over all slices
  def run_fine(self, u0):
    self.set_initial_value(u0)
    for i in range(0,self.nslices):
      # Run fine method
      self.slices[i].update_fine()
      # Fetch fine value and set initial value of next slice
      if i<self.nslices-1:
        self.set_initial_value( self.get_fine_value(i), i+1 )

  #
  # SET functions
  #
  def set_initial_value(self, u0, slice_nr=0):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    self.slices[slice_nr].set_sol_start(u0)

  #
  # GET functions
  #
  def get_coarse_value(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    return self.slices[slice_nr].get_sol_coarse()

  def get_fine_value(self, slice_nr):
    assert slice_nr<self.nslices, ("There are only %2i slices in this timemesh" % slice_nr)
    return self.slices[slice_nr].get_sol_fine()

  def get_fine_matrix(self, u0):
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

  def get_coarse_matrix(self, u0):
    Id = sparse.eye( u0.ndof, format="csc" )
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
