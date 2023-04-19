def position_hwf(self, evec, dir, hwf_evec=False, basis="orbital"):
    r"""

    Returns eigenvalues and optionally eigenvectors of the
    position operator matrix :math:`X` in either Bloch or orbital
    basis.  These eigenvectors can be interpreted as linear
    combinations of Bloch states *evec* that have minimal extent (or
    spread :math:`\Omega` in the sense of maximally localized
    Wannier functions) along direction *dir*. The eigenvalues are
    average positions of these localized states.

    Note that these eigenvectors are not maximally localized
    Wannier functions in the usual sense because they are
    localized only along one direction.  They are also not the
    average positions of the Bloch states *evec*, which are
    instead computed by :func:`pythtb.tb_model.position_expectation`.

    See function :func:`pythtb.tb_model.position_matrix` for
    the definition of the matrix :math:`X`.

    See also Fig. 3 in Phys. Rev. Lett. 102, 107603 (2009) for a
    discussion of the hybrid Wannier function centers in the
    context of a Chern insulator.

    :param evec: Eigenvectors for which we are computing matrix
      elements of the position operator.  The shape of this array
      is evec[band,orbital] if *nspin* equals 1 and
      evec[band,orbital,spin] if *nspin* equals 2.

    :param dir: Direction along which we are computing matrix
      elements.  This integer must not be one of the periodic
      directions since position operator matrix element in that
      case is not well defined.

    :param hwf_evec: Optional boolean variable.  If set to *True*
      this function will return not only eigenvalues but also
      eigenvectors of :math:`X`. Default value is *False*.

    :param basis: Optional parameter. If basis="bloch" then hybrid
      Wannier function *hwf_evec* is written in the Bloch basis.  I.e.
      hwf[i,j] correspond to the weight of j-th Bloch state from *evec*
      in the i-th hybrid Wannier function.  If basis="orbital" and nspin=1 then
      hwf[i,orb] correspond to the weight of orb-th orbital in the i-th
      hybrid Wannier function.  If basis="orbital" and nspin=2 then
      hwf[i,orb,spin] correspond to the weight of orb-th orbital, spin-th
      spin component in the i-th hybrid Wannier function.  Default value
      is "orbital".

    :returns:
      * **hwfc** -- Eigenvalues of the position operator matrix :math:`X`
        (also called hybrid Wannier function centers).
        Length of this vector equals number of bands given in *evec* input
        array.  Hybrid Wannier function centers are ordered in ascending order.
        Note that in general *n*-th hwfc does not correspond to *n*-th electronic
        state *evec*.

      * **hwf** -- Eigenvectors of the position operator matrix :math:`X`.
        (also called hybrid Wannier functions).  These are returned only if
        parameter *hwf_evec* is set to *True*.
        The shape of this array is [h,x] or [h,x,s] depending on value of *basis*
        and *nspin*.  If *basis* is "bloch" then x refers to indices of
        Bloch states *evec*.  If *basis* is "orbital" then *x* (or *x* and *s*)
        correspond to orbital index (or orbital and spin index if *nspin* is 2).

    Example usage::

      # diagonalizes Hamiltonian at some k-points
      (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
      # computes hybrid Wannier centers (and functions) for 3-rd kpoint
      # and bottom five bands along first coordinate
      (hwfc, hwf) = my_model.position_hwf(evecs[:5,2], 0, hwf_evec=True, basis="orbital")

    See also this example: :ref:`haldane_hwf-example`,

    """
    # check if model came from w90
    if self._assume_position_operator_diagonal == False:
        _offdiag_approximation_warning_and_stop()

    # get position matrix
    pos_mat = self.position_matrix(evec, dir)

    # diagonalize
    if hwf_evec == False:
        hwfc = np.linalg.eigvalsh(pos_mat)
        # sort eigenvalues and convert to real numbers
        hwfc = _nicefy_eig(hwfc)
        return np.array(hwfc, dtype=float)
    else:  # find eigenvalues and eigenvectors
        (hwfc, hwf) = np.linalg.eigh(pos_mat)
        # transpose matrix eig since otherwise it is confusing
        # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
        hwf = hwf.T
        # sort evectors, eigenvalues and convert to real numbers
        (hwfc, hwf) = _nicefy_eig(hwfc, hwf)
        # convert to right basis
        if basis.lower().strip() == "bloch":
            return (hwfc, hwf)
        elif basis.lower().strip() == "orbital":
            if self._nspin == 1:
                ret_hwf = np.zeros((hwf.shape[0], self._norb), dtype=complex)
                # sum over bloch states to get hwf in orbital basis
                for i in range(ret_hwf.shape[0]):
                    ret_hwf[i] = np.dot(hwf[i], evec)
                hwf = ret_hwf
            else:
                ret_hwf = np.zeros((hwf.shape[0], self._norb * 2), dtype=complex)
                # get rid of spin indices
                evec_use = evec.reshape([hwf.shape[0], self._norb * 2])
                # sum over states
                for i in range(ret_hwf.shape[0]):
                    ret_hwf[i] = np.dot(hwf[i], evec_use)
                # restore spin indices
                hwf = ret_hwf.reshape([hwf.shape[0], self._norb, 2])
            return (hwfc, hwf)
        else:
            raise Exception("\n\nBasis must be either bloch or orbital!")