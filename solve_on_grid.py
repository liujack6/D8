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