class tb_model(object):
    r"""
    This is the main class of the PythTB package which contains all
    information for the tight-binding model.

    :param dim_k: Dimensionality of reciprocal space, i.e., specifies how
      many directions are considered to be periodic.

    :param dim_r: Dimensionality of real space, i.e., specifies how many
      real space lattice vectors there are and how many coordinates are
      needed to specify the orbital coordinates.
    .. note:: Parameter *dim_r* can be larger than *dim_k*! For example,
      a polymer is a three-dimensional molecule (one needs three
      coordinates to specify orbital positions), but it is periodic
      along only one direction. For a polymer, therefore, we should
      have *dim_k* equal to 1 and *dim_r* equal to 3. See similar example
      here: :ref:`trestle-example`.

    :param lat: Array containing lattice vectors in Cartesian
      coordinates (in arbitrary units). In example the below, the first
      lattice vector has coordinates [1.0,0.5] while the second
      one has coordinates [0.0,2.0].  By default, lattice vectors
      are an identity matrix.

    :param orb: Array containing reduced coordinates of all
      tight-binding orbitals. In the example below, the first
      orbital is defined with reduced coordinates [0.2,0.3]. Its
      Cartesian coordinates are therefore 0.2 times the first
      lattice vector plus 0.3 times the second lattice vector.
      If *orb* is an integer code will assume that there are these many
      orbitals all at the origin of the unit cell.  By default
      the code will assume a single orbital at the origin.

    :param per: This is an optional parameter giving a list of lattice
      vectors which are considered to be periodic. In the example below,
      only the vector [0.0,2.0] is considered to be periodic (since
      per=[1]). By default, all lattice vectors are assumed to be
      periodic. If dim_k is smaller than dim_r, then by default the first
      dim_k vectors are considered to be periodic.

    :param nspin: Number of explicit spin components assumed for each
      orbital in *orb*. Allowed values of *nspin* are *1* and *2*. If
      *nspin* is 1 then the model is spinless, if *nspin* is 2 then it
      is explicitly a spinfull model and each orbital is assumed to
      have two spin components. Default value of this parameter is
      *1*.  Of course one can make spinfull calculation even with
      *nspin* set to 1, but then the user must keep track of which
      orbital corresponds to which spin component.

    Example usage::

       # Creates model that is two-dimensional in real space but only
       # one-dimensional in reciprocal space. Second lattice vector is
       # chosen to be periodic (since per=[1]). Three orbital
       # coordinates are specified.
       tb = tb_model(1, 2,
                   lat=[[1.0, 0.5], [0.0, 2.0]],
                   orb=[[0.2, 0.3], [0.1, 0.1], [0.2, 0.2]],
                   per=[1])

    """

    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):

        # initialize _dim_k = dimensionality of k-space (integer)
        if type(dim_k).__name__ != 'int':
            raise Exception("\n\nArgument dim_k not an integer")
        if dim_k < 0 or dim_k > 4:
            raise Exception("\n\nArgument dim_k out of range. Must be between 0 and 4.")
        self._dim_k = dim_k

        # initialize _dim_r = dimensionality of r-space (integer)
        if type(dim_r).__name__ != 'int':
            raise Exception("\n\nArgument dim_r not an integer")
        if dim_r < dim_k or dim_r > 4:
            raise Exception("\n\nArgument dim_r out of range. Must be dim_r>=dim_k and dim_r<=4.")
        self._dim_r = dim_r

        # initialize _lat = lattice vectors, array of dim_r*dim_r
        #   format is _lat(lat_vec_index,cartesian_index)
        # special option: 'unit' implies unit matrix, also default value
        if lat is 'unit' or lat is None:
            self._lat = np.identity(dim_r, float)
            print(" Lattice vectors not specified! I will use identity matrix.")
        elif type(lat).__name__ not in ['list', 'ndarray']:
            raise Exception("\n\nArgument lat is not a list.")
        else:
            self._lat = np.array(lat, dtype=float)
            if self._lat.shape != (dim_r, dim_r):
                raise Exception("\n\nWrong lat array dimensions")
        # check that volume is not zero and that have right handed system
        if dim_r > 0:
            if np.abs(np.linalg.det(self._lat)) < 1.0E-6:
                raise Exception("\n\nLattice vectors length/area/volume too close to zero, or zero.")
            if np.linalg.det(self._lat) < 0.0:
                raise Exception("\n\nLattice vectors need to form right handed system.")

        # initialize _norb = number of basis orbitals per cell
        #   and       _orb = orbital locations, in reduced coordinates
        #   format is _orb(orb_index,lat_vec_index)
        # special option: 'bravais' implies one atom at origin
        if orb is 'bravais' or orb is None:
            self._norb = 1
            self._orb = np.zeros((1, dim_r))
            print(" Orbital positions not specified. I will assume a single orbital at the origin.")
        elif type(orb).__name__ == 'int':
            self._norb = orb
            self._orb = np.zeros((orb, dim_r))
            print(" Orbital positions not specified. I will assume ", orb, " orbitals at the origin")
        elif type(orb).__name__ not in ['list', 'ndarray']:
            raise Exception("\n\nArgument orb is not a list or an integer")
        else:
            self._orb = np.array(orb, dtype=float)
            if len(self._orb.shape) != 2:
                raise Exception("\n\nWrong orb array rank")
            self._norb = self._orb.shape[0]  # number of orbitals
            if self._orb.shape[1] != dim_r:
                raise Exception("\n\nWrong orb array dimensions")

        # choose which self._dim_k out of self._dim_r dimensions are
        # to be considered periodic.
        if per == None:
            # by default first _dim_k dimensions are periodic
            self._per = list(range(self._dim_k))
        else:
            if len(per) != self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per = per

        # remember number of spin components
        if nspin not in [1, 2]:
            raise Exception("\n\nWrong value of nspin, must be 1 or 2!")
        self._nspin = nspin

        # by default, assume model did not come from w90 object and that
        # position operator is diagonal
        self._assume_position_operator_diagonal = True

        # compute number of electronic states at each k-point
        self._nsta = self._norb * self._nspin

        # Initialize onsite energies to zero
        if self._nspin == 1:
            self._site_energies = np.zeros((self._norb), dtype=float)
        elif self._nspin == 2:
            self._site_energies = np.zeros((self._norb, 2, 2), dtype=complex)
        # remember which onsite energies user has specified
        self._site_energies_specified = np.zeros(self._norb, dtype=bool)
        self._site_energies_specified[:] = False

        # Initialize hoppings to empty list
        self._hoppings = []

        # The onsite energies and hoppings are not specified
        # when creating a 'tb_model' object.  They are speficied
        # subsequently by separate function calls defined below.

    def set_onsite(self, onsite_en, ind_i=None, mode="set"):
        r"""
        Defines on-site energies for tight-binding orbitals. One can
        either set energy for one tight-binding orbital, or all at
        once.

        .. warning:: In previous version of PythTB this function was
          called *set_sites*. For backwards compatibility one can still
          use that name but that feature will be removed in future
          releases.

        :param onsite_en: Either a list of on-site energies (in
          arbitrary units) for each orbital, or a single on-site
          energy (in this case *ind_i* parameter must be given). In
          the case when *nspin* is *1* (spinless) then each on-site
          energy is a single number.  If *nspin* is *2* then on-site
          energy can be given either as a single number, or as an
          array of four numbers, or 2x2 matrix. If a single number is
          given, it is interpreted as on-site energy for both up and
          down spin component. If an array of four numbers is given,
          these are the coefficients of I, sigma_x, sigma_y, and
          sigma_z (that is, the 2x2 identity and the three Pauli spin
          matrices) respectively. Finally, full 2x2 matrix can be
          given as well. If this function is never called, on-site
          energy is assumed to be zero.

        :param ind_i: Index of tight-binding orbital whose on-site
          energy you wish to change. This parameter should be
          specified only when *onsite_en* is a single number (not a
          list).

        :param mode: Similar to parameter *mode* in function set_hop*.
          Speficies way in which parameter *onsite_en* is
          used. It can either set value of on-site energy from scratch,
          reset it, or add to it.

          * "set" -- Default value. On-site energy is set to value of
            *onsite_en* parameter. One can use "set" on each
            tight-binding orbital only once.

          * "reset" -- Specifies on-site energy to given value. This
            function can be called multiple times for the same
            orbital(s).

          * "add" -- Adds to the previous value of on-site
            energy. This function can be called multiple times for the
            same orbital(s).

        Example usage::

          # Defines on-site energy of first orbital to be 0.0,
          # second 1.0, and third 2.0
          tb.set_onsite([0.0, 1.0, 2.0])
          # Increases value of on-site energy for second orbital
          tb.set_onsite(100.0, 1, mode="add")
          # Changes on-site energy of second orbital to zero
          tb.set_onsite(0.0, 1, mode="reset")
          # Sets all three on-site energies at once
          tb.set_onsite([2.0, 3.0, 4.0], mode="reset")

        """
        if ind_i == None:
            if (len(onsite_en) != self._norb):
                raise Exception("\n\nWrong number of site energies")
        # make sure ind_i is not out of scope
        if ind_i != None:
            if ind_i < 0 or ind_i >= self._norb:
                raise Exception("\n\nIndex ind_i out of scope.")
        # make sure that onsite terms are real/hermitian
        if ind_i != None:
            to_check = [onsite_en]
        else:
            to_check = onsite_en
        for ons in to_check:
            if np.array(ons).shape == ():
                if np.abs(np.array(ons) - np.array(ons).conjugate()) > 1.0E-8:
                    raise Exception("\n\nOnsite energy should not have imaginary part!")
            elif np.array(ons).shape == (4,):
                if np.max(np.abs(np.array(ons) - np.array(ons).conjugate())) > 1.0E-8:
                    raise Exception("\n\nOnsite energy or Zeeman field should not have imaginary part!")
            elif np.array(ons).shape == (2, 2):
                if np.max(np.abs(np.array(ons) - np.array(ons).T.conjugate())) > 1.0E-8:
                    raise Exception("\n\nOnsite matrix should be Hermitian!")
        # specifying onsite energies from scratch, can be called only once
        if mode.lower() == "set":
            # specifying only one site at a time
            if ind_i != None:
                # make sure we specify things only once
                if self._site_energies_specified[ind_i] == True:
                    raise Exception(
                        "\n\nOnsite energy for this site was already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                    self._site_energies[ind_i] = self._val_to_block(onsite_en)
                    self._site_energies_specified[ind_i] = True
            # specifying all sites at once
            else:
                # make sure we specify things only once
                if True in self._site_energies_specified[ind_i]:
                    raise Exception(
                        "\n\nSome or all onsite energies were already specified! Use mode=\"reset\" or mode=\"add\".")
                else:
                    for i in range(self._norb):
                        self._site_energies[i] = self._val_to_block(onsite_en[i])
                    self._site_energies_specified[:] = True
        # reset values of onsite terms, without adding to previous value
        elif mode.lower() == "reset":
            # specifying only one site at a time
            if ind_i != None:
                self._site_energies[ind_i] = self._val_to_block(onsite_en)
                self._site_energies_specified[ind_i] = True
            # specifying all sites at once
            else:
                for i in range(self._norb):
                    self._site_energies[i] = self._val_to_block(onsite_en[i])
                self._site_energies_specified[:] = True
        # add to previous value
        elif mode.lower() == "add":
            # specifying only one site at a time
            if ind_i != None:
                self._site_energies[ind_i] += self._val_to_block(onsite_en)
                self._site_energies_specified[ind_i] = True
            # specifying all sites at once
            else:
                for i in range(self._norb):
                    self._site_energies[i] += self._val_to_block(onsite_en[i])
                self._site_energies_specified[:] = True
        else:
            raise Exception("\n\nWrong value of mode parameter")

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

    def _val_to_block(self, val):
        """If nspin=2 then returns a 2 by 2 matrix from the input
        parameters. If only one real number is given in the input then
        assume that this is the diagonal term. If array with four
        elements is given then first one is the diagonal term, and
        other three are Zeeman field direction. If given a 2 by 2
        matrix, just return it.  If nspin=1 then just returns val."""
        # spinless case
        if self._nspin == 1:
            return val
        # spinfull case
        elif self._nspin == 2:
            # matrix to return
            ret = np.zeros((2, 2), dtype=complex)
            #
            use_val = np.array(val)
            # only one number is given
            if use_val.shape == ():
                ret[0, 0] += use_val
                ret[1, 1] += use_val
            # if four numbers are given
            elif use_val.shape == (4,):
                # diagonal
                ret[0, 0] += use_val[0]
                ret[1, 1] += use_val[0]
                # sigma_x
                ret[0, 1] += use_val[1]
                ret[1, 0] += use_val[1]  # sigma_y
                ret[0, 1] += use_val[2] * (-1.0j)
                ret[1, 0] += use_val[2] * (1.0j)
                # sigma_z
                ret[0, 0] += use_val[3]
                ret[1, 1] += use_val[3] * (-1.0)
                # if 2 by 2 matrix is given
            elif use_val.shape == (2, 2):
                return use_val
            else:
                raise Exception( \
                    """\n
                    Wrong format of the on-site or hopping term. Must be single number, or
                    in the case of a spinfull model can be array of four numbers or 2x2
                    matrix.""")
            return ret

    def display(self):
        r"""
        Prints on the screen some information about this tight-binding
        model. This function doesn't take any parameters.
        """
        print('---------------------------------------')
        print('report of tight-binding model')
        print('---------------------------------------')
        print('k-space dimension           =', self._dim_k)
        print('r-space dimension           =', self._dim_r)
        print('number of spin components   =', self._nspin)
        print('periodic directions         =', self._per)
        print('number of orbitals          =', self._norb)
        print('number of electronic states =', self._nsta)
        print('lattice vectors:')
        for i, o in enumerate(self._lat):
            print(" #", _nice_int(i, 2), " ===>  [", end=' ')
            for j, v in enumerate(o):
                print(_nice_float(v, 7, 4), end=' ')
                if j != len(o) - 1:
                    print(",", end=' ')
            print("]")
        print('positions of orbitals:')
        for i, o in enumerate(self._orb):
            print(" #", _nice_int(i, 2), " ===>  [", end=' ')
            for j, v in enumerate(o):
                print(_nice_float(v, 7, 4), end=' ')
                if j != len(o) - 1:
                    print(",", end=' ')
            print("]")
        print('site energies:')
        for i, site in enumerate(self._site_energies):
            print(" #", _nice_int(i, 2), " ===>  ", end=' ')
            if self._nspin == 1:
                print(_nice_float(site, 7, 4))
            elif self._nspin == 2:
                print(str(site).replace("\n", " "))
        print('hoppings:')
        for i, hopping in enumerate(self._hoppings):
            print("<", _nice_int(hopping[1], 2), "| H |", _nice_int(hopping[2], 2), end=' ')
            if len(hopping) == 4:
                print("+ [", end=' ')
                for j, v in enumerate(hopping[3]):
                    print(_nice_int(v, 2), end=' ')
                    if j != len(hopping[3]) - 1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(">     ===> ", end=' ')
            if self._nspin == 1:
                print(_nice_complex(hopping[0], 7, 4))
            elif self._nspin == 2:
                print(str(hopping[0]).replace("\n", " "))
        print('hopping distances:')
        for i, hopping in enumerate(self._hoppings):
            print("|  pos(", _nice_int(hopping[1], 2), ")  - pos(", _nice_int(hopping[2], 2), end=' ')
            if len(hopping) == 4:
                print("+ [", end=' ')
                for j, v in enumerate(hopping[3]):
                    print(_nice_int(v, 2), end=' ')
                    if j != len(hopping[3]) - 1:
                        print(",", end=' ')
                    else:
                        print("]", end=' ')
            print(") |  =  ", end=' ')
            pos_i = np.dot(self._orb[hopping[1]], self._lat)
            pos_j = np.dot(self._orb[hopping[2]], self._lat)
            if len(hopping) == 4:
                pos_j += np.dot(hopping[3], self._lat)
            dist = np.linalg.norm(pos_j - pos_i)
            print(_nice_float(dist, 7, 4))

        print()

    def visualize(self, dir_first, dir_second=None, eig_dr=None, draw_hoppings=True, ph_color="black"):
        r"""

        Rudimentary function for visualizing tight-binding model geometry,
        hopping between tight-binding orbitals, and electron eigenstates.

        If eigenvector is not drawn, then orbitals in home cell are drawn
        as red circles, and those in neighboring cells are drawn with
        different shade of red. Hopping term directions are drawn with
        green lines connecting two orbitals. Origin of unit cell is
        indicated with blue dot, while real space unit vectors are drawn
        with blue lines.

        If eigenvector is drawn, then electron eigenstate on each orbital
        is drawn with a circle whose size is proportional to wavefunction
        amplitude while its color depends on the phase. There are various
        coloring schemes for the phase factor; see more details under
        *ph_color* parameter. If eigenvector is drawn and coloring scheme
        is "red-blue" or "wheel", all other elements of the picture are
        drawn in gray or black.

        :param dir_first: First index of Cartesian coordinates used for
          plotting.

        :param dir_second: Second index of Cartesian coordinates used for
          plotting. For example if dir_first=0 and dir_second=2, and
          Cartesian coordinates of some orbital is [2.0,4.0,6.0] then it
          will be drawn at coordinate [2.0,6.0]. If dimensionality of real
          space (*dim_r*) is zero or one then dir_second should not be
          specified.

        :param eig_dr: Optional parameter specifying eigenstate to
          plot. If specified, this should be one-dimensional array of
          complex numbers specifying wavefunction at each orbital in
          the tight-binding basis. If not specified, eigenstate is not
          drawn.

        :param draw_hoppings: Optional parameter specifying whether to
          draw all allowed hopping terms in the tight-binding
          model. Default value is True.

        :param ph_color: Optional parameter determining the way
          eigenvector phase factors are translated into color. Default
          value is "black". Convention of the wavefunction phase is as
          in convention 1 in section 3.1 of :download:`notes on
          tight-binding formalism  <misc/pythtb-formalism.pdf>`.  In
          other words, these wavefunction phases are in correspondence
          with cell-periodic functions :math:`u_{n {\bf k}} ({\bf r})`
          not :math:`\Psi_{n {\bf k}} ({\bf r})`.

          * "black" -- phase of eigenvectors are ignored and wavefunction
            is always colored in black.

          * "red-blue" -- zero phase is drawn red, while phases or pi or
            -pi are drawn blue. Phases in between are interpolated between
            red and blue. Some phase information is lost in this coloring
            becase phase of +phi and -phi have same color.

          * "wheel" -- each phase is given unique color. In steps of pi/3
            starting from 0, colors are assigned (in increasing hue) as:
            red, yellow, green, cyan, blue, magenta, red.

        :returns:
          * **fig** -- Figure object from matplotlib.pyplot module
            that can be used to save the figure in PDF, EPS or similar
            format, for example using fig.savefig("name.pdf") command.
          * **ax** -- Axes object from matplotlib.pyplot module that can be
            used to tweak the plot, for example by adding a plot title
            ax.set_title("Title goes here").

        Example usage::

          # Draws x-y projection of tight-binding model
          # tweaks figure and saves it as a PDF.
          (fig, ax) = tb.visualize(0, 1)
          ax.set_title("Title goes here")
          fig.savefig("model.pdf")

        See also these examples: :ref:`edge-example`,
        :ref:`visualize-example`.

        """

        # check the format of eig_dr
        if not (eig_dr is None):
            if eig_dr.shape != (self._norb,):
                raise Exception("\n\nWrong format of eig_dr! Must be array of size norb.")

        # check that ph_color is correct
        if ph_color not in ["black", "red-blue", "wheel"]:
            raise Exception("\n\nWrong value of ph_color parameter!")

        # check if dir_second had to be specified
        if dir_second == None and self._dim_r > 1:
            raise Exception("\n\nNeed to specify index of second coordinate for projection!")

        # start a new figure
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=[plt.rcParams["figure.figsize"][0],
                                  plt.rcParams["figure.figsize"][0]])
        ax = fig.add_subplot(111, aspect='equal')

        def proj(v):
            "Project vector onto drawing plane"
            coord_x = v[dir_first]
            if dir_second == None:
                coord_y = 0.0
            else:
                coord_y = v[dir_second]
            return [coord_x, coord_y]

        def to_cart(red):
            "Convert reduced to Cartesian coordinates"
            return np.dot(red, self._lat)

        # define colors to be used in plotting everything
        # except eigenvectors
        if (eig_dr is None) or ph_color == "black":
            c_cell = "b"
            c_orb = "r"
            c_nei = [0.85, 0.65, 0.65]
            c_hop = "g"
        else:
            c_cell = [0.4, 0.4, 0.4]
            c_orb = [0.0, 0.0, 0.0]
            c_nei = [0.6, 0.6, 0.6]
            c_hop = [0.0, 0.0, 0.0]

        # determine color scheme for eigenvectors
        def color_to_phase(ph):
            if ph_color == "black":
                return "k"
            if ph_color == "red-blue":
                ph = np.abs(ph / np.pi)
                return [1.0 - ph, 0.0, ph]
            if ph_color == "wheel":
                if ph < 0.0:
                    ph = ph + 2.0 * np.pi
                ph = 6.0 * ph / (2.0 * np.pi)
                x_ph = 1.0 - np.abs(ph % 2.0 - 1.0)
                if ph >= 0.0 and ph < 1.0: ret_col = [1.0, x_ph, 0.0]
                if ph >= 1.0 and ph < 2.0: ret_col = [x_ph, 1.0, 0.0]
                if ph >= 2.0 and ph < 3.0: ret_col = [0.0, 1.0, x_ph]
                if ph >= 3.0 and ph < 4.0: ret_col = [0.0, x_ph, 1.0]
                if ph >= 4.0 and ph < 5.0: ret_col = [x_ph, 0.0, 1.0]
                if ph >= 5.0 and ph <= 6.0: ret_col = [1.0, 0.0, x_ph]
                return ret_col

        # draw origin
        ax.plot([0.0], [0.0], "o", c=c_cell, mec="w", mew=0.0, zorder=7, ms=4.5)

        # first draw unit cell vectors which are considered to be periodic
        for i in self._per:
            # pick a unit cell vector and project it down to the drawing plane
            vec = proj(self._lat[i])
            ax.plot([0.0, vec[0]], [0.0, vec[1]], "-", c=c_cell, lw=1.5, zorder=7)

        # now draw all orbitals
        for i in range(self._norb):
            # find position of orbital in cartesian coordinates
            pos = to_cart(self._orb[i])
            pos = proj(pos)
            ax.plot([pos[0]], [pos[1]], "o", c=c_orb, mec="w", mew=0.0, zorder=10, ms=4.0)

        # draw hopping terms
        if draw_hoppings == True:
            for h in self._hoppings:
                # draw both i->j+R and i-R->j hop
                for s in range(2):
                    # get "from" and "to" coordinates
                    pos_i = np.copy(self._orb[h[1]])
                    pos_j = np.copy(self._orb[h[2]])
                    # add also lattice vector if not 0-dim
                    if self._dim_k != 0:
                        if s == 0:
                            pos_j[self._per] = pos_j[self._per] + h[3][self._per]
                        if s == 1:
                            pos_i[self._per] = pos_i[self._per] - h[3][self._per]
                    # project down vector to the plane
                    pos_i = np.array(proj(to_cart(pos_i)))
                    pos_j = np.array(proj(to_cart(pos_j)))
                    # add also one point in the middle to bend the curve
                    prcnt = 0.05  # bend always by this ammount
                    pos_mid = (pos_i + pos_j) * 0.5
                    dif = pos_j - pos_i  # difference vector
                    orth = np.array([dif[1], -1.0 * dif[0]])  # orthogonal to difference vector
                    orth = orth / np.sqrt(np.dot(orth, orth))  # normalize
                    pos_mid = pos_mid + orth * prcnt * np.sqrt(
                        np.dot(dif, dif))  # shift mid point in orthogonal direction
                    # draw hopping
                    all_pnts = np.array([pos_i, pos_mid, pos_j]).T
                    ax.plot(all_pnts[0], all_pnts[1], "-", c=c_hop, lw=0.75, zorder=8)
                    # draw "from" and "to" sites
                    ax.plot([pos_i[0]], [pos_i[1]], "o", c=c_nei, zorder=9, mew=0.0, ms=4.0, mec="w")
                    ax.plot([pos_j[0]], [pos_j[1]], "o", c=c_nei, zorder=9, mew=0.0, ms=4.0, mec="w")

        # now draw the eigenstate
        if not (eig_dr is None):
            for i in range(self._norb):
                # find position of orbital in cartesian coordinates
                pos = to_cart(self._orb[i])
                pos = proj(pos)
                # find norm of eigenfunction at this point
                nrm = (eig_dr[i] * eig_dr[i].conjugate()).real
                # rescale and get size of circle
                nrm_rad = 2.0 * nrm * float(self._norb)
                # get color based on the phase of the eigenstate
                phase = np.angle(eig_dr[i])
                c_ph = color_to_phase(phase)
                ax.plot([pos[0]], [pos[1]], "o", c=c_ph, mec="w", mew=0.0, ms=nrm_rad, zorder=11, alpha=0.8)

        # center the image
        #  first get the current limit, which is probably tight
        xl = ax.set_xlim()
        yl = ax.set_ylim()
        # now get the center of current limit
        centx = (xl[1] + xl[0]) * 0.5
        centy = (yl[1] + yl[0]) * 0.5
        # now get the maximal size (lengthwise or heightwise)
        mx = max([xl[1] - xl[0], yl[1] - yl[0]])
        # set new limits
        extr = 0.05  # add some boundary as well
        ax.set_xlim(centx - mx * (0.5 + extr), centx + mx * (0.5 + extr))
        ax.set_ylim(centy - mx * (0.5 + extr), centy + mx * (0.5 + extr))

        # return a figure and axes to the user
        return (fig, ax)

    def get_num_orbitals(self):
        "Returns number of orbitals in the model."
        return self._norb

    def get_orb(self):
        "Returns reduced coordinates of orbitals in format [orbital,coordinate.]"
        return self._orb.copy()

    def get_lat(self):
        "Returns lattice vectors in format [vector,coordinate]."
        return self._lat.copy()

    def _gen_ham(self, k_input=None):
        """Generate Hamiltonian for a certain k-point,
        K-point is given in reduced coordinates!"""
        kpnt = np.array(k_input)
        if not (k_input is None):
            # if kpnt is just a number then convert it to an array
            if len(kpnt.shape) == 0:
                kpnt = np.array([kpnt])
            # check that k-vector is of corect size
            if kpnt.shape != (self._dim_k,):
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self._dim_k != 0:
                raise Exception("\n\nHave to provide a k-vector!")
        # zero the Hamiltonian matrix
        if self._nspin == 1:
            ham = np.zeros((self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
            ham = np.zeros((self._norb, 2, self._norb, 2), dtype=complex)
        # modify diagonal elements
        for i in range(self._norb):
            if self._nspin == 1:
                ham[i, i] = self._site_energies[i]
            elif self._nspin == 2:
                ham[i, :, i, :] = self._site_energies[i]
        # go over all hoppings
        for hopping in self._hoppings:
            # get all data for the hopping parameter
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)
            i = hopping[1]
            j = hopping[2]
            # in 0-dim case there is no phase factor
            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)
                # vector from one site to another
                rv = -self._orb[i, :] + self._orb[j, :] + ind_R
                # Take only components of vector which are periodic
                rv = rv[self._per]
                # Calculate the hopping, see details in info/tb/tb.pdf
                phase = np.exp((2.0j) * np.pi * np.dot(kpnt, rv))
                amp = amp * phase
            # add this hopping into a matrix and also its conjugate
            if self._nspin == 1:
                ham[i, j] += amp
                ham[j, i] += amp.conjugate()
            elif self._nspin == 2:
                ham[i, :, j, :] += amp
                ham[j, :, i, :] += amp.T.conjugate()
        return ham

    def _sol_ham(self, ham, eig_vectors=False):
        """Solves Hamiltonian and returns eigenvectors, eigenvalues"""
        # reshape matrix first
        if self._nspin == 1:
            ham_use = ham
        elif self._nspin == 2:
            ham_use = ham.reshape((2 * self._norb, 2 * self._norb))
        # check that matrix is hermitian
        if np.max(ham_use - ham_use.T.conj()) > 1.0E-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        # solve matrix
        if eig_vectors == False:  # only find eigenvalues
            eval = np.linalg.eigvalsh(ham_use)
            # sort eigenvalues and convert to real numbers
            eval = _nicefy_eig(eval)
            return np.array(eval, dtype=float)
        else:  # find eigenvalues and eigenvectors
            (eval, eig) = np.linalg.eigh(ham_use)
            # transpose matrix eig since otherwise it is confusing
            # now eig[i,:] is eigenvector for eval[i]-th eigenvalue
            eig = eig.T
            # sort evectors, eigenvalues and convert to real numbers
            (eval, eig) = _nicefy_eig(eval, eig)
            # reshape eigenvectors if doing a spinfull calculation
            if self._nspin == 2:
                eig = eig.reshape((self._nsta, self._norb, 2))
            return (eval, eig)

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

    def solve_one(self, k_point=None, eig_vectors=False):
        r"""

        Similar to :func:`pythtb.tb_model.solve_all` but solves tight-binding
        model for only one k-vector.

        """
        # if not 0-dim case
        if not (k_point is None):
            if eig_vectors == False:
                eval = self.solve_all([k_point], eig_vectors=eig_vectors)
                # indices of eval are [band]
                return eval[:, 0]
            else:
                (eval, evec) = self.solve_all([k_point], eig_vectors=eig_vectors)
                # indices of eval are [band] for evec are [band,orbital,spin]
                if self._nspin == 1:
                    return (eval[:, 0], evec[:, 0, :])
                elif self._nspin == 2:
                    return (eval[:, 0], evec[:, 0, :, :])
        else:
            # do the same as solve_all
            return self.solve_all(eig_vectors=eig_vectors)

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

    def reduce_dim(self, remove_k, value_k):
        r"""
        Reduces dimensionality of the model by taking a reciprocal-space
        slice of the Bloch Hamiltonian :math:`{\cal H}_{\bf k}`. The Bloch
        Hamiltonian (defined in :download:`notes on tight-binding
        formalism <misc/pythtb-formalism.pdf>` in section 3.1 equation 3.7) of a
        d-dimensional model is a function of d-dimensional k-vector.

        This function returns a d-1 dimensional tight-binding model obtained
        by constraining one of k-vector components in :math:`{\cal H}_{\bf
        k}` to be a constant.

        :param remove_k: Which reciprocal space unit vector component
          you wish to keep constant.

        :param value_k: Value of the k-vector component to which you are
          constraining this model. Must be given in reduced coordinates.

        :returns:
          * **red_tb** -- Object of type :class:`pythtb.tb_model`
            representing a reduced tight-binding model.

        Example usage::

          # Constrains second k-vector component to equal 0.3
          red_tb = tb.reduce_dim(1, 0.3)

        """
        #
        if self._dim_k == 0:
            raise Exception("\n\nCan not reduce dimensionality even further!")
        # make a copy
        red_tb = copy.deepcopy(self)
        # make one of the directions not periodic
        red_tb._per.remove(remove_k)
        red_tb._dim_k = len(red_tb._per)
        # check that really removed one and only one direction
        if red_tb._dim_k != self._dim_k - 1:
            raise Exception("\n\nSpecified wrong dimension to reduce!")

        # specify hopping terms from scratch
        red_tb._hoppings = []
        # set all hopping parameters for this value of value_k
        for h in range(len(self._hoppings)):
            hop = self._hoppings[h]
            if self._nspin == 1:
                amp = complex(hop[0])
            elif self._nspin == 2:
                amp = np.array(hop[0], dtype=complex)
            i = hop[1];
            j = hop[2]
            ind_R = np.array(hop[3], dtype=int)
            # vector from one site to another
            rv = -red_tb._orb[i, :] + red_tb._orb[j, :] + np.array(ind_R, dtype=float)
            # take only r-vector component along direction you are not making periodic
            rv = rv[remove_k]
            # Calculate the part of hopping phase, only for this direction
            phase = np.exp((2.0j) * np.pi * (value_k * rv))
            # store modified version of the hop
            # Since we are getting rid of one dimension, it could be that now
            # one of the hopping terms became onsite term because one direction
            # is no longer periodic
            if i == j and (False not in (np.array(ind_R[red_tb._per], dtype=int) == 0)):
                if ind_R[remove_k] == 0:
                    # in this case this is really an onsite term
                    red_tb.set_onsite(amp * phase, i, mode="add")
                else:
                    # in this case must treat both R and -R because that term would
                    # have been counted twice without dimensional reduction
                    if self._nspin == 1:
                        red_tb.set_onsite(amp * phase + (amp * phase).conj(), i, mode="add")
                    elif self._nspin == 2:
                        red_tb.set_onsite(amp * phase + (amp.T * phase).conj(), i, mode="add")
            else:
                # just in case make the R vector zero along the reduction dimension
                ind_R[remove_k] = 0
                # add hopping term
                red_tb.set_hop(amp * phase, i, j, ind_R, mode="add", allow_conjugate_pair=True)

        return red_tb

    def make_supercell(self, sc_red_lat, return_sc_vectors=False, to_home=True):
        r"""

        Returns tight-binding model :class:`pythtb.tb_model`
        representing a super-cell of a current object. This function
        can be used together with *cut_piece* in order to create slabs
        with arbitrary surfaces.

        By default all orbitals will be shifted to the home cell after
        unit cell has been created. That way all orbitals will have
        reduced coordinates between 0 and 1. If you wish to avoid this
        behavior, you need to set, *to_home* argument to *False*.

        :param sc_red_lat: Array of integers with size *dim_r*dim_r*
          defining a super-cell lattice vectors in terms of reduced
          coordinates of the original tight-binding model. First index
          in the array specifies super-cell vector, while second index
          specifies coordinate of that super-cell vector.  If
          *dim_k<dim_r* then still need to specify full array with
          size *dim_r*dim_r* for consistency, but non-periodic
          directions must have 0 on off-diagonal elemets s and 1 on
          diagonal.

        :param return_sc_vectors: Optional parameter. Default value is
          *False*. If *True* returns also lattice vectors inside the
          super-cell. Internally, super-cell tight-binding model will
          have orbitals repeated in the same order in which these
          super-cell vectors are given, but if argument *to_home*
          is set *True* (which it is by default) then additionally,
          orbitals will be shifted to the home cell.

        :param to_home: Optional parameter, if *True* will
          shift all orbitals to the home cell. Default value is *True*.

        :returns:
          * **sc_tb** -- Object of type :class:`pythtb.tb_model`
            representing a tight-binding model in a super-cell.

          * **sc_vectors** -- Super-cell vectors, returned only if
            *return_sc_vectors* is set to *True* (default value is
            *False*).

        Example usage::

          # Creates super-cell out of 2d tight-binding model tb
          sc_tb = tb.make_supercell([[2, 1], [-1, 2]])

        """

        # Can't make super cell for model without periodic directions
        if self._dim_r == 0:
            raise Exception("\n\nMust have at least one periodic direction to make a super-cell")

        # convert array to numpy array
        use_sc_red_lat = np.array(sc_red_lat)

        # checks on super-lattice array
        if use_sc_red_lat.shape != (self._dim_r, self._dim_r):
            raise Exception("\n\nDimension of sc_red_lat array must be dim_r*dim_r")
        if use_sc_red_lat.dtype != int:
            raise Exception("\n\nsc_red_lat array elements must be integers")
        for i in range(self._dim_r):
            for j in range(self._dim_r):
                if (i == j) and (i not in self._per) and use_sc_red_lat[i, j] != 1:
                    raise Exception("\n\nDiagonal elements of sc_red_lat for non-periodic directions must equal 1.")
                if (i != j) and ((i not in self._per) or (j not in self._per)) and use_sc_red_lat[i, j] != 0:
                    raise Exception("\n\nOff-diagonal elements of sc_red_lat for non-periodic directions must equal 0.")
        if np.abs(np.linalg.det(use_sc_red_lat)) < 1.0E-6:
            raise Exception("\n\nSuper-cell lattice vectors length/area/volume too close to zero, or zero.")
        if np.linalg.det(use_sc_red_lat) < 0.0:
            raise Exception("\n\nSuper-cell lattice vectors need to form right handed system.")

        # converts reduced vector in original lattice to reduced vector in super-cell lattice
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(np.array(use_sc_red_lat.T, dtype=float),
                                   np.array(red_vec_orig, dtype=float))

        # conservative estimate on range of search for super-cell vectors
        max_R = np.max(np.abs(use_sc_red_lat)) * self._dim_r

        # candidates for super-cell vectors
        # this is hard-coded and can be improved!
        sc_cands = []
        if self._dim_r == 1:
            for i in range(-max_R, max_R + 1):
                sc_cands.append(np.array([i]))
        elif self._dim_r == 2:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    sc_cands.append(np.array([i, j]))
        elif self._dim_r == 3:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    for k in range(-max_R, max_R + 1):
                        sc_cands.append(np.array([i, j, k]))
        elif self._dim_r == 4:
            for i in range(-max_R, max_R + 1):
                for j in range(-max_R, max_R + 1):
                    for k in range(-max_R, max_R + 1):
                        for l in range(-max_R, max_R + 1):
                            sc_cands.append(np.array([i, j, k, l]))
        else:
            raise Exception("\n\nWrong dimensionality of dim_r!")

        # find all vectors inside super-cell
        # store them here
        sc_vec = []
        eps_shift = np.sqrt(2.0) * 1.0E-8  # shift of the grid, so to avoid double counting
        #
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red = to_red_sc(vec).tolist()
            # check if in the interior
            inside = True
            for t in tmp_red:
                if t <= -1.0 * eps_shift or t > 1.0 - eps_shift:
                    inside = False
            if inside == True:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc = len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(use_sc_red_lat)))) != num_sc:
            raise Exception("\n\nSuper-cell generation failed! Wrong number of super-cell vectors found.")

        # cartesian vectors of the super lattice
        sc_cart_lat = np.dot(use_sc_red_lat, self._lat)

        # orbitals of the super-cell tight-binding model
        sc_orb = []
        for cur_sc_vec in sc_vec:  # go over all super-cell vectors
            for orb in self._orb:  # go over all orbitals
                # shift orbital and compute coordinates in
                # reduced coordinates of super-cell
                sc_orb.append(to_red_sc(orb + cur_sc_vec))

        # create super-cell tb_model object to be returned
        sc_tb = tb_model(self._dim_k, self._dim_r, sc_cart_lat, sc_orb, per=self._per, nspin=self._nspin)

        # remember if came from w90
        sc_tb._assume_position_operator_diagonal = self._assume_position_operator_diagonal

        # repeat onsite energies
        for i in range(num_sc):
            for j in range(self._norb):
                sc_tb.set_onsite(self._site_energies[j], i * self._norb + j)

        # set hopping terms
        for c, cur_sc_vec in enumerate(sc_vec):  # go over all super-cell vectors
            for h in range(len(self._hoppings)):  # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp = self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R = copy.deepcopy(self._hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part = np.floor(to_red_sc(ind_R + cur_sc_vec))  # round down!
                sc_part = np.array(sc_part, dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part = ind_R + cur_sc_vec - np.dot(sc_part, use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind = None
                for p, pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec == orig_part):
                        if pair_ind != None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind = p
                if pair_ind == None:
                    raise Exception("\n\nDid not find super cell vector!")

                # index of "from" and "to" hopping indices
                hi = self._hoppings[h][1] + c * self._norb
                hj = self._hoppings[h][2] + pair_ind * self._norb

                # add hopping term
                sc_tb.set_hop(amp, hi, hj, sc_part, mode="add", allow_conjugate_pair=True)

        # put orbitals to home cell if asked for
        if to_home == True:
            sc_tb._shift_to_home()

        # return new tb model and vectors if needed
        if return_sc_vectors == False:
            return sc_tb
        else:
            return (sc_tb, sc_vec)

    def _shift_to_home(self):
        """Shifts all orbital positions to the home unit cell. After
        this function is called all reduced coordiantes of orbitals
        will be between 0 and 1. It may be useful to call this
        function after using make_supercell."""

        # go over all orbitals
        for i in range(self._norb):
            cur_orb = self._orb[i]
            # compute orbital in the home cell
            round_orb = (np.array(cur_orb) + 1.0E-6) % 1.0
            # find displacement vector needed to bring back to home cell
            disp_vec = np.array(np.round(cur_orb - round_orb), dtype=int)
            # check if have at least one non-zero component
            if True in (disp_vec != 0):
                # shift orbital
                self._orb[i] -= np.array(disp_vec, dtype=float)
                # shift also hoppings
                if self._dim_k != 0:
                    for h in range(len(self._hoppings)):
                        if self._hoppings[h][1] == i:
                            self._hoppings[h][3] -= disp_vec
                        if self._hoppings[h][2] == i:
                            self._hoppings[h][3] += disp_vec

    def remove_orb(self, to_remove):
        r"""

        Returns a model with some orbitals removed.  Note that this
        will reindex the orbitals with indices higher than those that
        are removed.  For example.  If model has 6 orbitals and one
        wants to remove 2nd orbital, then returned model will have 5
        orbitals indexed as 0,1,2,3,4.  In the returned model orbital
        indexed as 2 corresponds to the one indexed as 3 in the
        original model.  Similarly 3 and 4 correspond to 4 and 5.
        Indices of first two orbitals (0 and 1) are unaffected.

        :param to_remove: List of orbital indices to be removed, or
          index of single orbital to be removed

        :returns:

          * **del_tb** -- Object of type :class:`pythtb.tb_model`
            representing a model with removed orbitals.

        Example usage::

          # if original_model has say 10 orbitals then
          # returned small_model will have only 8 orbitals.

          small_model=original_model.remove_orb([2,5])

        """

        # if a single integer is given, convert to a list with one element
        if type(to_remove).__name__ == 'int':
            orb_index = [to_remove]
        else:
            orb_index = copy.deepcopy(to_remove)

        # check range of indices
        for i, orb_ind in enumerate(orb_index):
            if orb_ind < 0 or orb_ind > self._norb - 1 or type(orb_ind).__name__ != 'int':
                raise Exception("\n\nSpecified wrong orbitals to remove!")
        for i, ind1 in enumerate(orb_index):
            for ind2 in orb_index[i + 1:]:
                if ind1 == ind2:
                    raise Exception("\n\nSpecified duplicate orbitals to remove!")

        # put the orbitals to be removed in desceding order
        orb_index = sorted(orb_index, reverse=True)

        # make copy of a model
        ret = copy.deepcopy(self)

        # adjust some variables in the new model
        ret._norb -= len(orb_index)
        ret._nsta -= len(orb_index) * self._nspin
        # remove indices one by one
        for i, orb_ind in enumerate(orb_index):
            # adjust variables
            ret._orb = np.delete(ret._orb, orb_ind, 0)
            ret._site_energies = np.delete(ret._site_energies, orb_ind, 0)
            ret._site_energies_specified = np.delete(ret._site_energies_specified, orb_ind)
            # adjust hopping terms (in reverse)
            for j in range(len(ret._hoppings) - 1, -1, -1):
                h = ret._hoppings[j]
                # remove all terms that involve this orbital
                if h[1] == orb_ind or h[2] == orb_ind:
                    del ret._hoppings[j]
                else:  # otherwise modify term
                    if h[1] > orb_ind:
                        ret._hoppings[j][1] -= 1
                    if h[2] > orb_ind:
                        ret._hoppings[j][2] -= 1
        # return new model
        return ret

    def k_uniform_mesh(self, mesh_size):
        r"""
        Returns a uniform grid of k-points that can be passed to
        passed to function :func:`pythtb.tb_model.solve_all`.  This
        function is useful for plotting density of states histogram
        and similar.

        Returned uniform grid of k-points always contains the origin.

        :param mesh_size: Number of k-points in the mesh in each
          periodic direction of the model.

        :returns:

          * **k_vec** -- Array of k-vectors on the mesh that can be
            directly passed to function  :func:`pythtb.tb_model.solve_all`.

        Example usage::

          # returns a 10x20x30 mesh of a tight binding model
          # with three periodic directions
          k_vec = my_model.k_uniform_mesh([10,20,30])
          # solve model on the uniform mesh
          my_model.solve_all(k_vec)

        """

        # get the mesh size and checks for consistency
        use_mesh = np.array(list(map(round, mesh_size)), dtype=int)
        if use_mesh.shape != (self._dim_k,):
            print(use_mesh.shape)
            raise Exception("\n\nIncorrect size of the specified k-mesh!")
        if np.min(use_mesh) <= 0:
            raise Exception("\n\nMesh must have positive non-zero number of elements.")

        # construct the mesh
        if self._dim_k == 1:
            # get a mesh
            k_vec = np.mgrid[0:use_mesh[0]]
            # normalize the mesh
            norm = np.tile(np.array(use_mesh, dtype=float), use_mesh)
            norm = norm.reshape(use_mesh.tolist() + [1])
            norm = norm.transpose([1, 0])
            k_vec = k_vec / norm
            # final reshape
            k_vec = k_vec.transpose([1, 0]).reshape([use_mesh[0], 1])
        elif self._dim_k == 2:
            # get a mesh
            k_vec = np.mgrid[0:use_mesh[0], 0:use_mesh[1]]
            # normalize the mesh
            norm = np.tile(np.array(use_mesh, dtype=float), use_mesh)
            norm = norm.reshape(use_mesh.tolist() + [2])
            norm = norm.transpose([2, 0, 1])
            k_vec = k_vec / norm
            # final reshape
            k_vec = k_vec.transpose([1, 2, 0]).reshape([use_mesh[0] * use_mesh[1], 2])
        elif self._dim_k == 3:
            # get a mesh
            k_vec = np.mgrid[0:use_mesh[0], 0:use_mesh[1], 0:use_mesh[2]]
            # normalize the mesh
            norm = np.tile(np.array(use_mesh, dtype=float), use_mesh)
            norm = norm.reshape(use_mesh.tolist() + [3])
            norm = norm.transpose([3, 0, 1, 2])
            k_vec = k_vec / norm
            # final reshape
            k_vec = k_vec.transpose([1, 2, 3, 0]).reshape([use_mesh[0] * use_mesh[1] * use_mesh[2], 3])
        else:
            raise Exception("\n\nUnsupported dim_k!")

        return k_vec

    def k_path(self, kpts, nk, report=True):
        r"""

        Interpolates a path in reciprocal space between specified
        k-points.  In 2D or 3D the k-path can consist of several
        straight segments connecting high-symmetry points ("nodes"),
        and the results can be used to plot the bands along this path.

        The interpolated path that is returned contains as
        equidistant k-points as possible.

        :param kpts: Array of k-vectors in reciprocal space between
          which interpolated path should be constructed. These
          k-vectors must be given in reduced coordinates.  As a
          special case, in 1D k-space kpts may be a string:

          * *"full"*  -- Implies  *[ 0.0, 0.5, 1.0]*  (full BZ)
          * *"fullc"* -- Implies  *[-0.5, 0.0, 0.5]*  (full BZ, centered)
          * *"half"*  -- Implies  *[ 0.0, 0.5]*  (half BZ)

        :param nk: Total number of k-points to be used in making the plot.

        :param report: Optional parameter specifying whether printout
          is desired (default is True).

        :returns:

          * **k_vec** -- Array of (nearly) equidistant interpolated
            k-points. The distance between the points is calculated in
            the Cartesian frame, however coordinates themselves are
            given in dimensionless reduced coordinates!  This is done
            so that this array can be directly passed to function
            :func:`pythtb.tb_model.solve_all`.

          * **k_dist** -- Array giving accumulated k-distance to each
            k-point in the path.  Unlike array *k_vec* this one has
            dimensions! (Units are defined here so that for an
            one-dimensional crystal with lattice constant equal to for
            example *10* the length of the Brillouin zone would equal
            *1/10=0.1*.  In other words factors of :math:`2\pi` are
            absorbed into *k*.) This array can be used to plot path in
            the k-space so that the distances between the k-points in
            the plot are exact.

          * **k_node** -- Array giving accumulated k-distance to each
            node on the path in Cartesian coordinates.  This array is
            typically used to plot nodes (typically special points) on
            the path in k-space.

        Example usage::

          # Construct a path connecting four nodal points in k-space
          # Path will contain 401 k-points, roughly equally spaced
          path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
          (k_vec,k_dist,k_node) = my_model.k_path(path,401)
          # solve for eigenvalues on that path
          evals = tb.solve_all(k_vec)
          # then use evals, k_dist, and k_node to plot bandstructure
          # (see examples)

        """

        # processing of special cases for kpts
        if kpts == 'full':
            # full Brillouin zone for 1D case
            k_list = np.array([[0.], [0.5], [1.]])
        elif kpts == 'fullc':
            # centered full Brillouin zone for 1D case
            k_list = np.array([[-0.5], [0.], [0.5]])
        elif kpts == 'half':
            # half Brillouin zone for 1D case
            k_list = np.array([[0.], [0.5]])
        else:
            k_list = np.array(kpts)

        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape) == 1 and self._dim_k == 1:
            k_list = np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1] != self._dim_k:
            print('input k-space dimension is', k_list.shape[1])
            print('k-space dimension taken from model is', self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk < k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes = k_list.shape[0]

        # extract the lattice vectors from the TB model
        lat_per = np.copy(self._lat)
        # choose only those that correspond to periodic directions
        lat_per = lat_per[self._per]
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per, lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node = np.zeros(n_nodes, dtype=float)
        for n in range(1, n_nodes):
            dk = k_list[n] - k_list[n - 1]
            dklen = np.sqrt(np.dot(dk, np.dot(k_metric, dk)))
            k_node[n] = k_node[n - 1] + dklen

        # Find indices of nodes in interpolated list
        node_index = [0]
        for n in range(1, n_nodes - 1):
            frac = k_node[n] / k_node[-1]
            node_index.append(int(round(frac * (nk - 1))))
        node_index.append(nk - 1)

        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist = np.zeros(nk, dtype=float)
        #   array listing the interpolated k-points
        k_vec = np.zeros((nk, self._dim_k), dtype=float)

        # go over all kpoints
        k_vec[0] = k_list[0]
        for n in range(1, n_nodes):
            n_i = node_index[n - 1]
            n_f = node_index[n]
            kd_i = k_node[n - 1]
            kd_f = k_node[n]
            k_i = k_list[n - 1]
            k_f = k_list[n]
            for j in range(n_i, n_f + 1):
                frac = float(j - n_i) / float(n_f - n_i)
                k_dist[j] = kd_i + frac * (kd_f - kd_i)
                k_vec[j] = k_i + frac * (k_f - k_i)

        if report == True:
            if self._dim_k == 1:
                print(' Path in 1D BZ defined by nodes at ' + str(k_list.flatten()))
            else:
                print('----- k_path report begin ----------')
                original = np.get_printoptions()
                np.set_printoptions(precision=5)
                print('real-space lattice vectors\n', lat_per)
                print('k-space metric tensor\n', k_metric)
                print('internal coordinates of nodes\n', k_list)
                if (lat_per.shape[0] == lat_per.shape[1]):
                    # lat_per is invertible
                    lat_per_inv = np.linalg.inv(lat_per).T
                    print('reciprocal-space lattice vectors\n', lat_per_inv)
                    # cartesian coordinates of nodes
                    kpts_cart = np.tensordot(k_list, lat_per_inv, axes=1)
                    print('cartesian coordinates of nodes\n', kpts_cart)
                print('list of segments:')
                for n in range(1, n_nodes):
                    dk = k_node[n] - k_node[n - 1]
                    dk_str = _nice_float(dk, 7, 5)
                    print('  length = ' + dk_str + '  from ', k_list[n - 1], ' to ', k_list[n])
                print('node distance list:', k_node)
                print('node index list:   ', np.array(node_index))
                np.set_printoptions(precision=original["precision"])
                print('----- k_path report end ------------')
            print()

        return (k_vec, k_dist, k_node)

    def ignore_position_operator_offdiagonal(self):
        """Call to this function enables one to approximately compute
        Berry-like objects from tight-binding models that were
        obtained from Wannier90."""
        self._assume_position_operator_diagonal = True

    def position_matrix(self, evec, dir):
        r"""

        Returns matrix elements of the position operator along
        direction *dir* for eigenvectors *evec* at a single k-point.
        Position operator is defined in reduced coordinates.

        The returned object :math:`X` is

        .. math::

          X_{m n {\bf k}}^{\alpha} = \langle u_{m {\bf k}} \vert
          r^{\alpha} \vert u_{n {\bf k}} \rangle

        Here :math:`r^{\alpha}` is the position operator along direction
        :math:`\alpha` that is selected by *dir*.

        :param evec: Eigenvectors for which we are computing matrix
          elements of the position operator.  The shape of this array
          is evec[band,orbital] if *nspin* equals 1 and
          evec[band,orbital,spin] if *nspin* equals 2.

        :param dir: Direction along which we are computing the center.
          This integer must not be one of the periodic directions
          since position operator matrix element in that case is not
          well defined.

        :returns:
          * **pos_mat** -- Position operator matrix :math:`X_{m n}` as defined
            above. This is a square matrix with size determined by number of bands
            given in *evec* input array.  First index of *pos_mat* corresponds to
            bra vector (*m*) and second index to ket (*n*).

        Example usage::

          # diagonalizes Hamiltonian at some k-points
          (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
          # computes position operator matrix elements for 3-rd kpoint
          # and bottom five bands along first coordinate
          pos_mat = my_model.position_matrix(evecs[:5,2], 0)

        See also this example: :ref:`haldane_hwf-example`,

        """

        # make sure specified direction is not periodic!
        if dir in self._per:
            raise Exception("Can not compute position matrix elements along periodic direction!")
        # make sure direction is not out of range
        if dir < 0 or dir >= self._dim_r:
            raise Exception("Direction out of range!")

        # check if model came from w90
        if self._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()

        # get coordinates of orbitals along the specified direction
        pos_tmp = self._orb[:, dir]
        # reshape arrays in the case of spinfull calculation
        if self._nspin == 2:
            # tile along spin direction if needed
            pos_use = np.tile(pos_tmp, (2, 1)).transpose().flatten()
            # also flatten the state along the spin index
            evec_use = evec.reshape((evec.shape[0], evec.shape[1] * evec.shape[2]))
        else:
            pos_use = pos_tmp
            evec_use = evec

        # position matrix elements
        pos_mat = np.zeros((evec_use.shape[0], evec_use.shape[0]), dtype=complex)
        # go over all bands
        for i in range(evec_use.shape[0]):
            for j in range(evec_use.shape[0]):
                pos_mat[i, j] = np.dot(evec_use[i].conj(), pos_use * evec_use[j])

        # make sure matrix is hermitian
        if np.max(pos_mat - pos_mat.T.conj()) > 1.0E-9:
            raise Exception("\n\n Position matrix is not hermitian?!")

        return pos_mat

    def position_expectation(self, evec, dir):
        r"""

        Returns diagonal matrix elements of the position operator.
        These elements :math:`X_{n n}` can be interpreted as an
        average position of n-th Bloch state *evec[n]* along
        direction *dir*.  Generally speaking these centers are *not*
        hybrid Wannier function centers (which are instead
        returned by :func:`pythtb.tb_model.position_hwf`).

        See function :func:`pythtb.tb_model.position_matrix` for
        definition of matrix :math:`X`.

        :param evec: Eigenvectors for which we are computing matrix
          elements of the position operator.  The shape of this array
          is evec[band,orbital] if *nspin* equals 1 and
          evec[band,orbital,spin] if *nspin* equals 2.

        :param dir: Direction along which we are computing matrix
          elements.  This integer must not be one of the periodic
          directions since position operator matrix element in that
          case is not well defined.

        :returns:
          * **pos_exp** -- Diagonal elements of the position operator matrix :math:`X`.
            Length of this vector is determined by number of bands given in *evec* input
            array.

        Example usage::

          # diagonalizes Hamiltonian at some k-points
          (evals, evecs) = my_model.solve_all(k_vec,eig_vectors=True)
          # computes average position for 3-rd kpoint
          # and bottom five bands along first coordinate
          pos_exp = my_model.position_expectation(evecs[:5,2], 0)

        See also this example: :ref:`haldane_hwf-example`.

        """

        # check if model came from w90
        if self._assume_position_operator_diagonal == False:
            _offdiag_approximation_warning_and_stop()

        pos_exp = self.position_matrix(evec, dir).diagonal()
        return np.array(np.real(pos_exp), dtype=float)

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