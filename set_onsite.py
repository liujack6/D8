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