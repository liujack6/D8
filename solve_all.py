def solve_all(self, k_list=None, eig_vectors=False):
    r"""
    Solves for eigenvalues and (optionally) eigenvectors of the
    tight-binding model on a given one-dimensional list of k-vectors.

    .. note::

       Eigenvectors (wavefunctions) returned by this
       function and used throughout the code are exclusively given
       in convention 1 as described in section 3.1 of
       :download:`notes on tight-binding formalism
       <misc/pythtb-formalism.pdf>`.  In other words, they
       are in correspondence with cell-periodic functions
       :math:`u_{n {\bf k}} ({\bf r})` not
       :math:`\Psi_{n {\bf k}} ({\bf r})`.

    .. note::

       In some cases class :class:`pythtb.wf_array` provides a more
       elegant way to deal with eigensolutions on a regular mesh of
       k-vectors.

    :param k_list: One-dimensional array of k-vectors. Each k-vector
      is given in reduced coordinates of the reciprocal space unit
      cell. For example, for real space unit cell vectors [1.0,0.0]
      and [0.0,2.0] and associated reciprocal space unit vectors
      [2.0*pi,0.0] and [0.0,pi], k-vector with reduced coordinates
      [0.25,0.25] corresponds to k-vector [0.5*pi,0.25*pi].
      Dimensionality of each vector must equal to the number of
      periodic directions (i.e. dimensionality of reciprocal space,
      *dim_k*).
      This parameter shouldn't be specified for system with
      zero-dimensional k-space (*dim_k* =0).

    :param eig_vectors: Optional boolean parameter, specifying whether
      eigenvectors should be returned. If *eig_vectors* is True, then
      both eigenvalues and eigenvectors are returned, otherwise only
      eigenvalues are returned.

    :returns:
      * **eval** -- Two dimensional array of eigenvalues for
        all bands for all kpoints. Format is eval[band,kpoint] where
        first index (band) corresponds to the electron band in
        question and second index (kpoint) corresponds to the k-point
        as listed in the input parameter *k_list*. Eigenvalues are
        sorted from smallest to largest at each k-point seperately.

        In the case when reciprocal space is zero-dimensional (as in a
        molecule) kpoint index is dropped and *eval* is of the format
        eval[band].

      * **evec** -- Three dimensional array of eigenvectors for
        all bands and all kpoints. If *nspin* equals 1 the format
        of *evec* is evec[band,kpoint,orbital] where "band" is the
        electron band in question, "kpoint" is index of k-vector
        as given in input parameter *k_list*. Finally, "orbital"
        refers to the tight-binding orbital basis function.
        Ordering of bands is the same as in *eval*.

        Eigenvectors evec[n,k,j] correspond to :math:`C^{n {\bf
        k}}_{j}` from section 3.1 equation 3.5 and 3.7 of the
        :download:`notes on tight-binding formalism
        <misc/pythtb-formalism.pdf>`.

        In the case when reciprocal space is zero-dimensional (as in a
        molecule) kpoint index is dropped and *evec* is of the format
        evec[band,orbital].

        In the spinfull calculation (*nspin* equals 2) evec has
        additional component evec[...,spin] corresponding to the
        spin component of the wavefunction.

    Example usage::

      # Returns eigenvalues for three k-vectors
      eval = tb.solve_all([[0.0, 0.0], [0.0, 0.2], [0.0, 0.5]])
      # Returns eigenvalues and eigenvectors for two k-vectors
      (eval, evec) = tb.solve_all([[0.0, 0.0], [0.0, 0.2]], eig_vectors=True)

    """
    # if not 0-dim case
    if not (k_list is None):
        nkp = len(k_list)  # number of k points
        # first initialize matrices for all return data
        #    indices are [band,kpoint]
        ret_eval = np.zeros((self._nsta, nkp), dtype=float)
        #    indices are [band,kpoint,orbital,spin]
        if self._nspin == 1:
            ret_evec = np.zeros((self._nsta, nkp, self._norb), dtype=complex)
        elif self._nspin == 2:
            ret_evec = np.zeros((self._nsta, nkp, self._norb, 2), dtype=complex)
        # go over all kpoints
        for i, k in enumerate(k_list):
            # generate Hamiltonian at that point
            ham = self._gen_ham(k)
            # solve Hamiltonian
            if eig_vectors == False:
                eval = self._sol_ham(ham, eig_vectors=eig_vectors)
                ret_eval[:, i] = eval[:]
            else:
                (eval, evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
                ret_eval[:, i] = eval[:]
                if self._nspin == 1:
                    ret_evec[:, i, :] = evec[:, :]
                elif self._nspin == 2:
                    ret_evec[:, i, :, :] = evec[:, :, :]
        # return stuff
        if eig_vectors == False:
            # indices of eval are [band,kpoint]
            return ret_eval
        else:
            # indices of eval are [band,kpoint] for evec are [band,kpoint,orbital,(spin)]
            return (ret_eval, ret_evec)
    else:  # 0 dim case
        # generate Hamiltonian
        ham = self._gen_ham()
        # solve
        if eig_vectors == False:
            eval = self._sol_ham(ham, eig_vectors=eig_vectors)
            # indices of eval are [band]
            return eval
        else:
            (eval, evec) = self._sol_ham(ham, eig_vectors=eig_vectors)
            # indices of eval are [band] and of evec are [band,orbital,spin]
            return (eval, evec)