def set_hop(self, hop_amp, ind_i, ind_j, ind_R=None, mode="set", allow_conjugate_pair=False):
    r"""

    Defines hopping parameters between tight-binding orbitals. In
    the notation used in section 3.1 equation 3.6 of
    :download:`notes on tight-binding formalism
    <misc/pythtb-formalism.pdf>` this function specifies the
    following object

    .. math::

      H_{ij}({\bf R})= \langle \phi_{{\bf 0} i}  \vert H  \vert \phi_{{\bf R},j} \rangle

    Where :math:`\langle \phi_{{\bf 0} i} \vert` is i-th
    tight-binding orbital in the home unit cell and
    :math:`\vert \phi_{{\bf R},j} \rangle` is j-th tight-binding orbital in
    unit cell shifted by lattice vector :math:`{\bf R}`. :math:`H`
    is the Hamiltonian.

    (Strictly speaking, this term specifies hopping amplitude
    for hopping from site *j+R* to site *i*, not vice-versa.)

    Hopping in the opposite direction is automatically included by
    the code since

    .. math::

      H_{ji}(-{\bf R})= \left[ H_{ij}({\bf R}) \right]^{*}

    .. warning::

       There is no need to specify hoppings in both :math:`i
       \rightarrow j+R` direction and opposite :math:`j
       \rightarrow i-R` direction since that is done
       automatically. If you want to specifiy hoppings in both
       directions, see description of parameter
       *allow_conjugate_pair*.

    .. warning:: In previous version of PythTB this function was
      called *add_hop*. For backwards compatibility one can still
      use that name but that feature will be removed in future
      releases.

    :param hop_amp: Hopping amplitude; can be real or complex
      number, equals :math:`H_{ij}({\bf R})`. If *nspin* is *2*
      then hopping amplitude can be given either as a single
      number, or as an array of four numbers, or as 2x2 matrix. If
      a single number is given, it is interpreted as hopping
      amplitude for both up and down spin component.  If an array
      of four numbers is given, these are the coefficients of I,
      sigma_x, sigma_y, and sigma_z (that is, the 2x2 identity and
      the three Pauli spin matrices) respectively. Finally, full
      2x2 matrix can be given as well.

    :param ind_i: Index of bra orbital from the bracket :math:`\langle
      \phi_{{\bf 0} i} \vert H \vert \phi_{{\bf R},j} \rangle`. This
      orbital is assumed to be in the home unit cell.

    :param ind_j: Index of ket orbital from the bracket :math:`\langle
      \phi_{{\bf 0} i} \vert H \vert \phi_{{\bf R},j} \rangle`. This
      orbital does not have to be in the home unit cell; its unit cell
      position is determined by parameter *ind_R*.

    :param ind_R: Specifies, in reduced coordinates, the shift of
      the ket orbital. The number of coordinates must equal the
      dimensionality in real space (*dim_r* parameter) for consistency,
      but only the periodic directions of ind_R will be considered. If
      reciprocal space is zero-dimensional (as in a molecule),
      this parameter does not need to be specified.

    :param mode: Similar to parameter *mode* in function *set_onsite*.
      Speficies way in which parameter *hop_amp* is
      used. It can either set value of hopping term from scratch,
      reset it, or add to it.

      * "set" -- Default value. Hopping term is set to value of
        *hop_amp* parameter. One can use "set" for each triplet of
        *ind_i*, *ind_j*, *ind_R* only once.

      * "reset" -- Specifies on-site energy to given value. This
        function can be called multiple times for the same triplet
        *ind_i*, *ind_j*, *ind_R*.

      * "add" -- Adds to the previous value of hopping term This
        function can be called multiple times for the same triplet
        *ind_i*, *ind_j*, *ind_R*.

      If *set_hop* was ever called with *allow_conjugate_pair* set
      to True, then it is possible that user has specified both
      :math:`i \rightarrow j+R` and conjugate pair :math:`j
      \rightarrow i-R`.  In this case, "set", "reset", and "add"
      parameters will treat triplet *ind_i*, *ind_j*, *ind_R* and
      conjugate triplet *ind_j*, *ind_i*, *-ind_R* as distinct.

    :param allow_conjugate_pair: Default value is *False*. If set
      to *True* code will allow user to specify hopping
      :math:`i \rightarrow j+R` even if conjugate-pair hopping
      :math:`j \rightarrow i-R` has been
      specified. If both terms are specified, code will
      still count each term two times.

    Example usage::

      # Specifies complex hopping amplitude between first orbital in home
      # unit cell and third orbital in neigbouring unit cell.
      tb.set_hop(0.3+0.4j, 0, 2, [0, 1])
      # change value of this hopping
      tb.set_hop(0.1+0.2j, 0, 2, [0, 1], mode="reset")
      # add to previous value (after this function call below,
      # hopping term amplitude is 100.1+0.2j)
      tb.set_hop(100.0, 0, 2, [0, 1], mode="add")

    """
    #
    if self._dim_k != 0 and (ind_R is None):
        raise Exception("\n\nNeed to specify ind_R!")
    # if necessary convert from integer to array
    if self._dim_k == 1 and type(ind_R).__name__ == 'int':
        tmpR = np.zeros(self._dim_r, dtype=int)
        tmpR[self._per] = ind_R
        ind_R = tmpR
    # check length of ind_R
    if self._dim_k != 0:
        if len(ind_R) != self._dim_r:
            raise Exception("\n\nLength of input ind_R vector must equal dim_r! Even if dim_k<dim_r.")
    # make sure ind_i and ind_j are not out of scope
    if ind_i < 0 or ind_i >= self._norb:
        raise Exception("\n\nIndex ind_i out of scope.")
    if ind_j < 0 or ind_j >= self._norb:
        raise Exception("\n\nIndex ind_j out of scope.")
        # do not allow onsite hoppings to be specified here because then they
    # will be double-counted
    if self._dim_k == 0:
        if ind_i == ind_j:
            raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
    else:
        if ind_i == ind_j:
            all_zer = True
            for k in self._per:
                if int(ind_R[k]) != 0:
                    all_zer = False
            if all_zer == True:
                raise Exception("\n\nDo not use set_hop for onsite terms. Use set_onsite instead!")
    #
    # make sure that if <i|H|j+R> is specified that <j|H|i-R> is not!
    if allow_conjugate_pair == False:
        for h in self._hoppings:
            if ind_i == h[2] and ind_j == h[1]:
                if self._dim_k == 0:
                    raise Exception( \
                        """\n
                        Following matrix element was already implicitely specified:
                           i=""" + str(ind_i) + " j=" + str(ind_j) + """
Remember, specifying <i|H|j> automatically specifies <j|H|i>.  For
consistency, specify all hoppings for a given bond in the same
direction.  (Or, alternatively, see the documentation on the
'allow_conjugate_pair' flag.)
""")
                elif False not in (np.array(ind_R)[self._per] == (-1) * np.array(h[3])[self._per]):
                    raise Exception( \
                        """\n
                        Following matrix element was already implicitely specified:
                           i=""" + str(ind_i) + " j=" + str(ind_j) + " R=" + str(ind_R) + """
Remember,specifying <i|H|j+R> automatically specifies <j|H|i-R>.  For
consistency, specify all hoppings for a given bond in the same
direction.  (Or, alternatively, see the documentation on the
'allow_conjugate_pair' flag.)
""")
    # convert to 2by2 matrix if needed
    hop_use = self._val_to_block(hop_amp)
    # hopping term parameters to be stored
    if self._dim_k == 0:
        new_hop = [hop_use, int(ind_i), int(ind_j)]
    else:
        new_hop = [hop_use, int(ind_i), int(ind_j), np.array(ind_R)]
    #
    # see if there is a hopping term with same i,j,R
    use_index = None
    for iih, h in enumerate(self._hoppings):
        # check if the same
        same_ijR = False
        if ind_i == h[1] and ind_j == h[2]:
            if self._dim_k == 0:
                same_ijR = True
            else:
                if False not in (np.array(ind_R)[self._per] == np.array(h[3])[self._per]):
                    same_ijR = True
        # if they are the same then store index of site at which they are the same
        if same_ijR == True:
            use_index = iih
    #
    # specifying hopping terms from scratch, can be called only once
    if mode.lower() == "set":
        # make sure we specify things only once
        if use_index != None:
            raise Exception(
                "\n\nHopping energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
        else:
            self._hoppings.append(new_hop)
    # reset value of hopping term, without adding to previous value
    elif mode.lower() == "reset":
        if use_index != None:
            self._hoppings[use_index] = new_hop
        else:
            self._hoppings.append(new_hop)
    # add to previous value
    elif mode.lower() == "add":
        if use_index != None:
            self._hoppings[use_index][0] += new_hop[0]
        else:
            self._hoppings.append(new_hop)
    else:
        raise Exception("\n\nWrong value of mode parameter")