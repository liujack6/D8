class wf_array(object):
    r"""

    This class is used to solve a tight-binding model
    :class:`pythtb.tb_model` on a regular or non-regular grid
    of points in reciprocal space and/or parameter space, and
    perform on it various calculations. For example it can be
    used to calculate the Berry phase, Berry curvature, 1st Chern
    number, etc.

    *Regular k-space grid*:
    If the grid is a regular k-mesh (no parametric dimensions),
    a single call to the function
    :func:`pythtb.wf_array.solve_on_grid` will both construct a
    k-mesh that uniformly covers the Brillouin zone, and populate
    it with wavefunctions (eigenvectors) computed on this grid.
    The last point in each k-dimension is set so that it represents
    the same Bloch function as the first one (this involves the
    insertion of some orbital-position-dependent phase factors).

    Example :ref:`haldane_bp-example` shows how to use wf_array on
    a regular grid of points in k-space. Examples :ref:`cone-example`
    and :ref:`3site_cycle-example` show how to use non-regular grid of
    points.

    *Parametric or irregular k-space grid grid*:
    An irregular grid of points, or a grid that includes also
    one or more parametric dimensions, can be populated manually
    with the help of the *[]* operator.  For example, to copy
    eigenvectors *evec* into coordinate (2,3) in the *wf_array*
    object *wf* one can simply do::

      wf[2,3]=evec

    The eigenvectors (wavefunctions) *evec* in the example above
    are expected to be in the format *evec[band,orbital]*
    (or *evec[band,orbital,spin]* for the spinfull calculation).
    This is the same format as returned by
    :func:`pythtb.tb_model.solve_one` or
    :func:`pythtb.tb_model.solve_all` (in the latter case one
    needs to restrict it to a single k-point as *evec[:,kpt,:]*
    if the model has *dim_k>=1*).

    If wf_array is used for closed paths, either in a
    reciprocal-space or parametric direction, then one needs to
    include both the starting and ending eigenfunctions even though
    they are physically equivalent.  If the array dimension in
    question is a k-vector direction and the path traverses the
    Brillouin zone in a primitive reciprocal-lattice direction,
    :func:`pythtb.wf_array.impose_pbc` can be used to associate
    the starting and ending points with each other; if it is a
    non-winding loop in k-space or a loop in parameter space,
    then :func:`pythtb.wf_array.impose_loop` can be used instead.
    (These may not be necessary if only Berry fluxes are needed.)

    Example :ref:`3site_cycle-example` shows how one
    of the directions of *wf_array* object need not be a k-vector
    direction, but can instead be a Hamiltonian parameter :math:`\lambda`
    (see also discussion after equation 4.1 in :download:`notes on
    tight-binding formalism <misc/pythtb-formalism.pdf>`).

    :param model: Object of type :class:`pythtb.tb_model` representing
      tight-binding model associated with this array of eigenvectors.

    :param mesh_arr: Array giving a dimension of the grid of points in
      each reciprocal-space or parametric direction.

    Example usage::

      # Construct wf_array capable of storing an 11x21 array of
      # wavefunctions
      wf = wf_array(tb, [11, 21])
      # populate this wf_array with regular grid of points in
      # Brillouin zone
      wf.solve_on_grid([0.0, 0.0])

      # Compute set of eigenvectors at one k-point
      (eval, evec) = tb.solve_one([kx, ky], eig_vectors = True)
      # Store it manually into a specified location in the array
      wf[3, 4] = evec
      # To access the eigenvectors from the same position
      print wf[3, 4]

    """

    def __init__(self, model, mesh_arr):
        # number of electronic states for each k-point
        self._nsta = model._nsta
        # number of spin components
        self._nspin = model._nspin
        # number of orbitals
        self._norb = model._norb
        # store orbitals from the model
        self._orb = np.copy(model._orb)
        # store entire model as well
        self._model = copy.deepcopy(model)
        # store dimension of array of points on which to keep wavefunctions
        self._mesh_arr = np.array(mesh_arr)
        self._dim_arr = len(self._mesh_arr)
        # all dimensions should be 2 or larger, because pbc can be used
        if True in (self._mesh_arr <= 1).tolist():
            raise Exception("\n\nDimension of wf_array object in each direction must be 2 or larger.")
        # generate temporary array used later to generate object ._wfs
        wfs_dim = np.copy(self._mesh_arr)
        wfs_dim = np.append(wfs_dim, self._nsta)
        wfs_dim = np.append(wfs_dim, self._norb)
        if self._nspin == 2:
            wfs_dim = np.append(wfs_dim, self._nspin)
            # store wavefunctions here in the form _wfs[kx_index,ky_index, ... ,band,orb,spin]
        self._wfs = np.zeros(wfs_dim, dtype=complex)

    def solve_on_grid(self, start_k):
        r"""

        Solve a tight-binding model on a regular mesh of k-points covering
        the entire reciprocal-space unit cell. Both points at the opposite
        sides of reciprocal-space unit cell are included in the array.

        This function also automatically imposes periodic boundary
        conditions on the eigenfunctions. See also the discussion in
        :func:`pythtb.wf_array.impose_pbc`.

        :param start_k: Origin of a regular grid of points in the reciprocal space.

        :returns:
          * **gaps** -- returns minimal direct bandgap between n-th and n+1-th
              band on all the k-points in the mesh.  Note that in the case of band
              crossings one may have to use very dense k-meshes to resolve
              the crossing.

        Example usage::

          # Solve eigenvectors on a regular grid anchored
          # at a given point
          wf.solve_on_grid([-0.5, -0.5])

        """
        # check dimensionality
        if self._dim_arr != self._model._dim_k:
            raise Exception(
                "\n\nIf using solve_on_grid method, dimension of wf_array must equal dim_k of the tight-binding model!")
        # to return gaps at all k-points
        if self._norb <= 1:
            all_gaps = None  # trivial case since there is only one band
        else:
            gap_dim = np.copy(self._mesh_arr) - 1
            gap_dim = np.append(gap_dim, self._norb * self._nspin - 1)
            all_gaps = np.zeros(gap_dim, dtype=float)
        #
        if self._dim_arr == 1:
            # don't need to go over the last point because that will be
            # computed in the impose_pbc call
            for i in range(self._mesh_arr[0] - 1):
                # generate a kpoint
                kpt = [start_k[0] + float(i) / float(self._mesh_arr[0] - 1)]
                # solve at that point
                (eval, evec) = self._model.solve_one(kpt, eig_vectors=True)
                # store wavefunctions
                self[i] = evec
                # store gaps
                if all_gaps is not None:
                    all_gaps[i, :] = eval[1:] - eval[:-1]
            # impose boundary conditions
            self.impose_pbc(0, self._model._per[0])
        elif self._dim_arr == 2:
            for i in range(self._mesh_arr[0] - 1):
                for j in range(self._mesh_arr[1] - 1):
                    kpt = [start_k[0] + float(i) / float(self._mesh_arr[0] - 1), \
                           start_k[1] + float(j) / float(self._mesh_arr[1] - 1)]
                    (eval, evec) = self._model.solve_one(kpt, eig_vectors=True)
                    self[i, j] = evec
                    if all_gaps is not None:
                        all_gaps[i, j, :] = eval[1:] - eval[:-1]
            for dir in range(2):
                self.impose_pbc(dir, self._model._per[dir])
        elif self._dim_arr == 3:
            for i in range(self._mesh_arr[0] - 1):
                for j in range(self._mesh_arr[1] - 1):
                    for k in range(self._mesh_arr[2] - 1):
                        kpt = [start_k[0] + float(i) / float(self._mesh_arr[0] - 1), \
                               start_k[1] + float(j) / float(self._mesh_arr[1] - 1), \
                               start_k[2] + float(k) / float(self._mesh_arr[2] - 1)]
                        (eval, evec) = self._model.solve_one(kpt, eig_vectors=True)
                        self[i, j, k] = evec
                        if all_gaps is not None:
                            all_gaps[i, j, k, :] = eval[1:] - eval[:-1]
            for dir in range(3):
                self.impose_pbc(dir, self._model._per[dir])
        elif self._dim_arr == 4:
            for i in range(self._mesh_arr[0] - 1):
                for j in range(self._mesh_arr[1] - 1):
                    for k in range(self._mesh_arr[2] - 1):
                        for l in range(self._mesh_arr[3] - 1):
                            kpt = [start_k[0] + float(i) / float(self._mesh_arr[0] - 1), \
                                   start_k[1] + float(j) / float(self._mesh_arr[1] - 1), \
                                   start_k[2] + float(k) / float(self._mesh_arr[2] - 1), \
                                   start_k[3] + float(l) / float(self._mesh_arr[3] - 1)]
                            (eval, evec) = self._model.solve_one(kpt, eig_vectors=True)
                            self[i, j, k, l] = evec
                            if all_gaps is not None:
                                all_gaps[i, j, k, l, :] = eval[1:] - eval[:-1]
            for dir in range(4):
                self.impose_pbc(dir, self._model._per[dir])
        else:
            raise Exception("\n\nWrong dimensionality!")

        return all_gaps.min(axis=tuple(range(self._dim_arr)))

    def __check_key(self, key):
        # do some checks for 1D
        if self._dim_arr == 1:
            if type(key).__name__ != 'int':
                raise TypeError("Key should be an integer!")
            if key < (-1) * self._mesh_arr[0] or key >= self._mesh_arr[0]:
                raise IndexError("Key outside the range!")
        # do checks for higher dimension
        else:
            if len(key) != self._dim_arr:
                raise TypeError("Wrong dimensionality of key!")
            for i, k in enumerate(key):
                if type(k).__name__ != 'int':
                    raise TypeError("Key should be set of integers!")
                if k < (-1) * self._mesh_arr[i] or k >= self._mesh_arr[i]:
                    raise IndexError("Key outside the range!")

    def __getitem__(self, key):
        # check that key is in the correct range
        self.__check_key(key)
        # return wavefunction
        return self._wfs[key]

    def __setitem__(self, key, value):
        # check that key is in the correct range
        self.__check_key(key)
        # store wavefunction
        self._wfs[key] = np.array(value, dtype=complex)

    def impose_pbc(self, mesh_dir, k_dir):
        r"""

        If the *wf_array* object was populated using the
        :func:`pythtb.wf_array.solve_on_grid` method, this function
        should not be used since it will be called automatically by
        the code.

        The eigenfunctions :math:`\Psi_{n {\bf k}}` are by convention
        chosen to obey a periodic gauge, i.e.,
        :math:`\Psi_{n,{\bf k+G}}=\Psi_{n {\bf k}}` not only up to a
        phase, but they are also equal in phase.  It follows that
        the cell-periodic Bloch functions are related by
        :math:`u_{n,{\bf k+G}}=e^{-i{\bf G}\cdot{\bf r}} u_{n {\bf k}}`.
        See :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>` section 4.4 and equation 4.18 for
        more detail.  This routine sets the cell-periodic Bloch function
        at the end of the string in direction :math:`{\bf G}` according
        to this formula, overwriting the previous value.

        This function will impose these periodic boundary conditions along
        one direction of the array. We are assuming that the k-point
        mesh increases by exactly one reciprocal lattice vector along
        this direction. This is currently **not** checked by the code;
        it is the responsibility of the user. Currently *wf_array*
        does not store the k-vectors on which the model was solved;
        it only stores the eigenvectors (wavefunctions).

        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        :param k_dir: Corresponding to the periodic k-vector direction
          in the Brillouin zone of the underlying *tb_model*.  Since
          version 1.7.0 this parameter is defined so that it is
          specified between 0 and *dim_r-1*.

        See example :ref:`3site_cycle-example`, where the periodic boundary
        condition is applied only along one direction of *wf_array*.

        Example usage::

          # Imposes periodic boundary conditions along the mesh_dir=0
          # direction of the wf_array object, assuming that along that
          # direction the k_dir=1 component of the k-vector is increased
          # by one reciprocal lattice vector.  This could happen, for
          # example, if the underlying tb_model is two dimensional but
          # wf_array is a one-dimensional path along k_y direction.
          wf.impose_pbc(mesh_dir=0,k_dir=1)

        """

        if k_dir not in self._model._per:
            raise Exception("Periodic boundary condition can be specified only along periodic directions!")

        # Compute phase factors
        ffac = np.exp(-2.j * np.pi * self._orb[:, k_dir])
        if self._nspin == 1:
            phase = ffac
        else:
            # for spinors, same phase multiplies both components
            phase = np.zeros((self._norb, 2), dtype=complex)
            phase[:, 0] = ffac
            phase[:, 1] = ffac

        # Copy first eigenvector onto last one, multiplying by phase factors
        # We can use numpy broadcasting since the orbital index is last
        if mesh_dir == 0:
            self._wfs[-1, ...] = self._wfs[0, ...] * phase
        elif mesh_dir == 1:
            self._wfs[:, -1, ...] = self._wfs[:, 0, ...] * phase
        elif mesh_dir == 2:
            self._wfs[:, :, -1, ...] = self._wfs[:, :, 0, ...] * phase
        elif mesh_dir == 3:
            self._wfs[:, :, :, -1, ...] = self._wfs[:, :, :, 0, ...] * phase
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def impose_loop(self, mesh_dir):
        r"""

        If the user knows that the first and last points along the
        *mesh_dir* direction correspond to the same Hamiltonian (this
        is **not** checked), then this routine can be used to set the
        eigenvectors equal (with equal phase), by replacing the last
        eigenvector with the first one (for each band, and for each
        other mesh direction, if any).

        This routine should not be used if the first and last points
        are related by a reciprocal lattice vector; in that case,
        :func:`pythtb.wf_array.impose_pbc` should be used instead.

        :param mesh_dir: Direction of wf_array along which you wish to
          impose periodic boundary conditions.

        Example usage::

          # Suppose the wf_array object is three-dimensional
          # corresponding to (kx,ky,lambda) where (kx,ky) are
          # wavevectors of a 2D insulator and lambda is an
          # adiabatic parameter that goes around a closed loop.
          # Then to insure that the states at the ends of the lambda
          # path are equal (with equal phase) in preparation for
          # computing Berry phases in lambda for given (kx,ky),
          # do wf.impose_loop(mesh_dir=2)

        """

        # Copy first eigenvector onto last one
        if mesh_dir == 0:
            self._wfs[-1, ...] = self._wfs[0, ...]
        elif mesh_dir == 1:
            self._wfs[:, -1, ...] = self._wfs[:, 0, ...]
        elif mesh_dir == 2:
            self._wfs[:, :, -1, ...] = self._wfs[:, :, 0, ...]
        elif mesh_dir == 3:
            self._wfs[:, :, :, -1, ...] = self._wfs[:, :, :, 0, ...]
        else:
            raise Exception("\n\nWrong value of mesh_dir.")

    def berry_phase(self, occ, dir=None, contin=True, berry_evals=False):
        r"""

        Computes the Berry phase along a given array direction and
        for a given set of occupied states.  This assumes that the
        occupied bands are well separated in energy from unoccupied
        bands. It is the responsibility of the user to check that
        this is satisfied.  By default, the Berry phase traced over
        occupied bands is returned, but optionally the individual
        phases of the eigenvalues of the global unitary rotation
        matrix (corresponding to "maximally localized Wannier
        centers" or "Wilson loop eigenvalues") can be requested
        (see parameter *berry_evals* for more details).

        For an array of size *N* in direction $dir$, the Berry phase
        is computed from the *N-1* inner products of neighboring
        eigenfunctions.  This corresponds to an "open-path Berry
        phase" if the first and last points have no special
        relation.  If they correspond to the same physical
        Hamiltonian, and have been properly aligned in phase using
        :func:`pythtb.wf_array.impose_pbc` or
        :func:`pythtb.wf_array.impose_loop`, then a closed-path
        Berry phase will be computed.

        For a one-dimensional wf_array (i.e., a single string), the
        computed Berry phases are always chosen to be between -pi and pi.
        For a higher dimensional wf_array, the Berry phase is computed
        for each one-dimensional string of points, and an array of
        Berry phases is returned. The Berry phase for the first string
        (with lowest index) is always constrained to be between -pi and
        pi. The range of the remaining phases depends on the value of
        the input parameter *contin*.

        The discretized formula used to compute Berry phase is described
        in Sec. 4.5 of :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>`.

        :param occ: Array of indices of energy bands which are considered
          to be occupied.

        :param dir: Index of wf_array direction along which Berry phase is
          computed. This parameters needs not be specified for
          a one-dimensional wf_array.

        :param contin: Optional boolean parameter. If True then the
          branch choice of the Berry phase (which is indeterminate
          modulo 2*pi) is made so that neighboring strings (in the
          direction of increasing index value) have as close as
          possible phases. The phase of the first string (with lowest
          index) is always constrained to be between -pi and pi. If
          False, the Berry phase for every string is constrained to be
          between -pi and pi. The default value is True.

        :param berry_evals: Optional boolean parameter. If True then
          will compute and return the phases of the eigenvalues of the
          product of overlap matrices. (These numbers correspond also
          to hybrid Wannier function centers.) These phases are either
          forced to be between -pi and pi (if *contin* is *False*) or
          they are made to be continuous (if *contin* is True).

        :returns:
          * **pha** -- If *berry_evals* is False (default value) then
            returns the Berry phase for each string. For a
            one-dimensional wf_array this is just one number. For a
            higher-dimensional wf_array *pha* contains one phase for
            each one-dimensional string in the following format. For
            example, if *wf_array* contains k-points on mesh with
            indices [i,j,k] and if direction along which Berry phase
            is computed is *dir=1* then *pha* will be two dimensional
            array with indices [i,k], since Berry phase is computed
            along second direction. If *berry_evals* is True then for
            each string returns phases of all eigenvalues of the
            product of overlap matrices. In the convention used for
            previous example, *pha* in this case would have indices
            [i,k,n] where *n* refers to index of individual phase of
            the product matrix eigenvalue.

        Example usage::

          # Computes Berry phases along second direction for three lowest
          # occupied states. For example, if wf is threedimensional, then
          # pha[2,3] would correspond to Berry phase of string of states
          # along wf[2,:,3]
          pha = wf.berry_phase([0, 1, 2], 1)

        See also these examples: :ref:`haldane_bp-example`,
        :ref:`cone-example`, :ref:`3site_cycle-example`,

        """

        # check if model came from w90
        if self._model._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()

        # if dir<0 or dir>self._dim_arr-1:
        #  raise Exception("\n\nDirection key out of range")
        #
        # This could be coded more efficiently, but it is hard-coded for now.
        #
        # 1D case
        if self._dim_arr == 1:
            # pick which wavefunctions to use
            wf_use = self._wfs[:, occ, :]
            # calculate berry phase
            ret = _one_berry_loop(wf_use, berry_evals)
        # 2D case
        elif self._dim_arr == 2:
            # choice along which direction you wish to calculate berry phase
            if dir == 0:
                ret = []
                for i in range(self._mesh_arr[1]):
                    wf_use = self._wfs[:, i, :, :][:, occ, :]
                    ret.append(_one_berry_loop(wf_use, berry_evals))
            elif dir == 1:
                ret = []
                for i in range(self._mesh_arr[0]):
                    wf_use = self._wfs[i, :, :, :][:, occ, :]
                    ret.append(_one_berry_loop(wf_use, berry_evals))
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        # 3D case
        elif self._dim_arr == 3:
            # choice along which direction you wish to calculate berry phase
            if dir == 0:
                ret = []
                for i in range(self._mesh_arr[1]):
                    ret_t = []
                    for j in range(self._mesh_arr[2]):
                        wf_use = self._wfs[:, i, j, :, :][:, occ, :]
                        ret_t.append(_one_berry_loop(wf_use, berry_evals))
                    ret.append(ret_t)
            elif dir == 1:
                ret = []
                for i in range(self._mesh_arr[0]):
                    ret_t = []
                    for j in range(self._mesh_arr[2]):
                        wf_use = self._wfs[i, :, j, :, :][:, occ, :]
                        ret_t.append(_one_berry_loop(wf_use, berry_evals))
                    ret.append(ret_t)
            elif dir == 2:
                ret = []
                for i in range(self._mesh_arr[0]):
                    ret_t = []
                    for j in range(self._mesh_arr[1]):
                        wf_use = self._wfs[i, j, :, :, :][:, occ, :]
                        ret_t.append(_one_berry_loop(wf_use, berry_evals))
                    ret.append(ret_t)
            else:
                raise Exception("\n\nWrong direction for Berry phase calculation!")
        else:
            raise Exception("\n\nWrong dimensionality!")

        # convert phases to numpy array
        if self._dim_arr > 1 or berry_evals == True:
            ret = np.array(ret, dtype=float)

        # make phases of eigenvalues continuous
        if contin == True:
            # iron out 2pi jumps, make the gauge choice such that first phase in the
            # list is fixed, others are then made continuous.
            if berry_evals == False:
                # 2D case
                if self._dim_arr == 2:
                    ret = _one_phase_cont(ret, ret[0])
                # 3D case
                elif self._dim_arr == 3:
                    for i in range(ret.shape[1]):
                        if i == 0:
                            clos = ret[0, 0]
                        else:
                            clos = ret[0, i - 1]
                        ret[:, i] = _one_phase_cont(ret[:, i], clos)
                elif self._dim_arr != 1:
                    raise Exception("\n\nWrong dimensionality!")
            # make eigenvalues continuous. This does not take care of band-character
            # at band crossing for example it will just connect pairs that are closest
            # at neighboring points.
            else:
                # 2D case
                if self._dim_arr == 2:
                    ret = _array_phases_cont(ret, ret[0, :])
                # 3D case
                elif self._dim_arr == 3:
                    for i in range(ret.shape[1]):
                        if i == 0:
                            clos = ret[0, 0, :]
                        else:
                            clos = ret[0, i - 1, :]
                        ret[:, i] = _array_phases_cont(ret[:, i], clos)
                elif self._dim_arr != 1:
                    raise Exception("\n\nWrong dimensionality!")
        return ret

    def position_matrix(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_matrix`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()
        #
        evec = self._wfs[tuple(key)][occ]
        return self._model.position_matrix(evec, dir)

    def position_expectation(self, key, occ, dir):
        """Similar to :func:`pythtb.tb_model.position_expectation`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()
        #
        evec = self._wfs[tuple(key)][occ]
        return self._model.position_expectation(evec, dir)

    def position_hwf(self, key, occ, dir, hwf_evec=False, basis="bloch"):
        """Similar to :func:`pythtb.tb_model.position_hwf`.  Only
        difference is that states are now specified with key in the
        mesh *key* and indices of bands *occ*."""
        # check if model came from w90
        if self._model._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()
        #
        evec = self._wfs[tuple(key)][occ]
        return self._model.position_hwf(evec, dir, hwf_evec, basis)

    def berry_flux(self, occ, dirs=None, individual_phases=False):
        r"""

        In the case of a 2-dimensional *wf_array* array calculates the
        integral of Berry curvature over the entire plane.  In higher
        dimensional case (3 or 4) it will compute integrated curvature
        over all 2-dimensional slices of a higher-dimensional
        *wf_array*.

        :param occ: Array of indices of energy bands which are considered
          to be occupied.

        :param dirs: Array of indices of two wf_array directions on which
          the Berry flux is computed. This parameter needs not be
          specified for a two-dimensional wf_array.  By default *dirs* takes
          first two directions in the array.

        :param individual_phases: If *True* then returns Berry phase
          for each plaquette (small square) in the array. Default
          value is *False*.

        :returns:

          * **flux** -- In a 2-dimensional case returns and integral
            of Berry curvature (if *individual_phases* is *True* then
            returns integral of Berry phase around each plaquette).
            In higher dimensional case returns integral of Berry
            curvature over all slices defined with directions *dirs*.
            Returned value is an array over the remaining indices of
            *wf_array*.  (If *individual_phases* is *True* then it
            returns again phases around each plaquette for each
            slice. First indices define the slice, last two indices
            index the plaquette.)

        Example usage::

          # Computes integral of Berry curvature of first three bands
          flux = wf.berry_flux([0, 1, 2])

        """

        # check if model came from w90
        if self._model._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()

        # default case is to take first two directions for flux calculation
        if dirs == None:
            dirs = [0, 1]

        # consistency checks
        if dirs[0] == dirs[1]:
            raise Exception("Need to specify two different directions for Berry flux calculation.")
        if dirs[0] >= self._dim_arr or dirs[1] >= self._dim_arr or dirs[0] < 0 or dirs[1] < 0:
            raise Exception("Direction for Berry flux calculation out of bounds.")

        # 2D case
        if self._dim_arr == 2:
            # compute the fluxes through all plaquettes on the entire plane
            ord = list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0] = dirs[0]
            ord[1] = dirs[1]
            plane_wfs = self._wfs.transpose(ord)
            # take bands of choice
            plane_wfs = plane_wfs[:, :, occ]

            # compute fluxes
            all_phases = _one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases == False:
                return all_phases.sum()
            else:
                return all_phases

        # 3D or 4D case
        elif self._dim_arr in [3, 4]:
            # compute the fluxes through all plaquettes on the entire plane
            ord = list(range(len(self._wfs.shape)))
            # select two directions from dirs
            ord[0] = dirs[0]
            ord[1] = dirs[1]

            # find directions over which we wish to loop
            ld = list(range(self._dim_arr))
            ld.remove(dirs[0])
            ld.remove(dirs[1])
            if len(ld) != self._dim_arr - 2:
                raise Exception("Hm, this should not happen? Inconsistency with the mesh size.")

            # add remaining indices
            if self._dim_arr == 3:
                ord[2] = ld[0]
            if self._dim_arr == 4:
                ord[2] = ld[0]
                ord[3] = ld[1]

            # reorder wavefunctions
            use_wfs = self._wfs.transpose(ord)

            # loop over the the remaining direction
            if self._dim_arr == 3:
                slice_phases = np.zeros(
                    (self._mesh_arr[ord[2]], self._mesh_arr[dirs[0]] - 1, self._mesh_arr[dirs[1]] - 1), dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    # take a 2d slice
                    plane_wfs = use_wfs[:, :, i]
                    # take bands of choice
                    plane_wfs = plane_wfs[:, :, occ]
                    # compute fluxes on the slice
                    slice_phases[i, :, :] = _one_flux_plane(plane_wfs)
            elif self._dim_arr == 4:
                slice_phases = np.zeros((self._mesh_arr[ord[2]], self._mesh_arr[ord[3]], self._mesh_arr[dirs[0]] - 1,
                                         self._mesh_arr[dirs[1]] - 1), dtype=float)
                for i in range(self._mesh_arr[ord[2]]):
                    for j in range(self._mesh_arr[ord[3]]):
                        # take a 2d slice
                        plane_wfs = use_wfs[:, :, i, j]
                        # take bands of choice
                        plane_wfs = plane_wfs[:, :, occ]
                        # compute fluxes on the slice
                        slice_phases[i, j, :, :] = _one_flux_plane(plane_wfs)

            # return either total flux or individual phase for each plaquete
            if individual_phases == False:
                return slice_phases.sum(axis=(-2, -1))
            else:
                return slice_phases

        else:
            raise Exception("\n\nWrong dimensionality!")

    def berry_curv(self, occ, individual_phases=False):
        r"""

      .. warning:: This function has been renamed as :func:`pythtb.berry_flux` and is provided
        here only for backwards compatibility with versions of pythtb prior to 1.7.0.  Please
        use related :func:`pythtb.berry_flux` as this function may not exist in future releases.

        """

        print(""" 

Warning:
  Usage of function berry_curv is discouraged.
  It has been renamed as berry_flux, which should be used instead.
""")
        return self.berry_flux(occ, individual_phases)