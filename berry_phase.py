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