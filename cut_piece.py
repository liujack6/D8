def cut_piece(self, num, fin_dir, glue_edgs=False):
    r"""
    Constructs a (d-1)-dimensional tight-binding model out of a
    d-dimensional one by repeating the unit cell a given number of
    times along one of the periodic lattice vectors. The real-space
    lattice vectors of the returned model are the same as those of
    the original model; only the dimensionality of reciprocal space
    is reduced.

    :param num: How many times to repeat the unit cell.

    :param fin_dir: Index of the real space lattice vector along
      which you no longer wish to maintain periodicity.

    :param glue_edgs: Optional boolean parameter specifying whether to
      allow hoppings from one edge to the other of a cut model.

    :returns:
      * **fin_model** -- Object of type
        :class:`pythtb.tb_model` representing a cutout
        tight-binding model. Orbitals in *fin_model* are
        numbered so that the i-th orbital of the n-th unit
        cell has index i+norb*n (here norb is the number of
        orbitals in the original model).

    Example usage::

      A = tb_model(3, 3, ...)
      # Construct two-dimensional model B out of three-dimensional
      # model A by repeating model along second lattice vector ten times
      B = A.cut_piece(10, 1)
      # Further cut two-dimensional model B into one-dimensional model
      # A by repeating unit cell twenty times along third lattice
      # vector and allow hoppings from one edge to the other
      C = B.cut_piece(20, 2, glue_edgs=True)

    See also these examples: :ref:`haldane_fin-example`,
    :ref:`edge-example`.


    """
    if self._dim_k == 0:
        raise Exception("\n\nModel is already finite")
    if type(num).__name__ != 'int':
        raise Exception("\n\nArgument num not an integer")

    # check value of num
    if num < 1:
        raise Exception("\n\nArgument num must be positive!")
    if num == 1 and glue_edgs == True:
        raise Exception("\n\nCan't have num==1 and glueing of the edges!")

    # generate orbitals of a finite model
    fin_orb = []
    onsite = []  # store also onsite energies
    for i in range(num):  # go over all cells in finite direction
        for j in range(self._norb):  # go over all orbitals in one cell
            # make a copy of j-th orbital
            orb_tmp = np.copy(self._orb[j, :])
            # change coordinate along finite direction
            orb_tmp[fin_dir] += float(i)
            # add to the list
            fin_orb.append(orb_tmp)
            # do the onsite energies at the same time
            onsite.append(self._site_energies[j])
    onsite = np.array(onsite)
    fin_orb = np.array(fin_orb)

    # generate periodic directions of a finite model
    fin_per = copy.deepcopy(self._per)
    # find if list of periodic directions contains the one you
    # want to make finite
    if fin_per.count(fin_dir) != 1:
        raise Exception("\n\nCan not make model finite along this direction!")
    # remove index which is no longer periodic
    fin_per.remove(fin_dir)

    # generate object of tb_model type that will correspond to a cutout
    fin_model = tb_model(self._dim_k - 1,
                         self._dim_r,
                         copy.deepcopy(self._lat),
                         fin_orb,
                         fin_per,
                         self._nspin)

    # remember if came from w90
    fin_model._assume_position_operator_diagonal = self._assume_position_operator_diagonal

    # now put all onsite terms for the finite model
    fin_model.set_onsite(onsite, mode="reset")

    # put all hopping terms
    for c in range(num):  # go over all cells in finite direction
        for h in range(len(self._hoppings)):  # go over all hoppings in one cell
            # amplitude of the hop is the same
            amp = self._hoppings[h][0]

            # lattice vector of the hopping
            ind_R = copy.deepcopy(self._hoppings[h][3])
            jump_fin = ind_R[fin_dir]  # store by how many cells is the hopping in finite direction
            if fin_model._dim_k != 0:
                ind_R[fin_dir] = 0  # one of the directions now becomes finite

            # index of "from" and "to" hopping indices
            hi = self._hoppings[h][1] + c * self._norb
            #   have to compensate  for the fact that ind_R in finite direction
            #   will not be used in the finite model
            hj = self._hoppings[h][2] + (c + jump_fin) * self._norb

            # decide whether this hopping should be added or not
            to_add = True
            # if edges are not glued then neglect all jumps that spill out
            if glue_edgs == False:
                if hj < 0 or hj >= self._norb * num:
                    to_add = False
            # if edges are glued then do mod division to wrap up the hopping
            else:
                hj = int(hj) % int(self._norb * num)

            # add hopping to a finite model
            if to_add == True:
                if fin_model._dim_k == 0:
                    fin_model.set_hop(amp, hi, hj, mode="add", allow_conjugate_pair=True)
                else:
                    fin_model.set_hop(amp, hi, hj, ind_R, mode="add", allow_conjugate_pair=True)

    return fin_model