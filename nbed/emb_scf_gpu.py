### copy of emb_scf.py
### but cupy replaces numpy! import cupy as np is the only change and
### get_mo_integrals are returned as numpy arrays instead of cupy arrays.

import cupy as np
from gpu4pyscf import scf, dft
from pyscf import gto
from pyscf import mcscf
import numpy

class EmbedSCF_GPU():

    def __init__(self, global_scf_obj,
                 act_MO_idxs, env_MO_idxs, 
                 mo_coeff,
                 mo_occ, 
                 Sao,
                 max_memory_MB, 
                 mu_val=1e6):
        
        self.mu_val = mu_val
        self.Sao = Sao

        if len(mo_occ.shape) != 1:
            raise ValueError("SCF input must be restricted (not unrestricted)")
        
        assert len(act_MO_idxs) + len(env_MO_idxs) == len(mo_occ), "active and environment MO indices must sum to total number of MOs"
        assert len(act_MO_idxs) + len(env_MO_idxs) == global_scf_obj.mol.nao, "active and environment MO indices must equal number of MOs"
        
        if np.any(mo_occ==1) or hasattr(global_scf_obj, "nelec"):
            ## hasattr is needed as sometimes a user may pass in a open-shell SCF even if it is restricted (aka all double occupied) 
            self.SCF_type = "open-shell"
        else:
            self.SCF_type = "closed-shell"

        self.global_scf_obj = global_scf_obj

        # use PySCF CAS order is [core_fixed, active, remaining_virtual]
        occ_all = np.where(mo_occ > 0)[0]
        vir_all = np.where(mo_occ == 0)[0]

        ## intersect, not setdiff: setdiff1d(occ_all, act_MO_idxs) is the *environment*
        ## occupied block, so naming it occ_act silently swaps the two subsystems and
        ## embeds the complement of the requested fragment
        occ_act   = np.intersect1d(occ_all, act_MO_idxs)
        vir_act   = np.intersect1d(vir_all, act_MO_idxs)
        occ_env   = np.intersect1d(occ_all, env_MO_idxs)
        vir_env   = np.intersect1d(vir_all, env_MO_idxs)

        re_idx = np.concatenate([occ_env, occ_act, vir_act, vir_env])
        assert len(np.setdiff1d(re_idx, np.arange(global_scf_obj.mol.nao))) == 0, "re_indexing wrong"
    

        self.ncore = len(occ_env) # frozen env orbitals
        self.n_act = len(act_MO_idxs) 
        self.act_cols = np.arange(self.ncore, self.ncore + self.n_act)
        self.remaining_cols = np.setdiff1d(np.arange(global_scf_obj.mol.nao), self.act_cols)

        self.non_core_idx_all = np.setdiff1d(np.arange(global_scf_obj.mol.nao), np.arange(self.ncore))
        ### note virtual env is included in remaining_cols! this is good for WF methods!

        self.C_full_reidx = mo_coeff[:, re_idx].copy()
        self.mo_occ_full_reidx = mo_occ[re_idx].copy()
        self.mo_occ_act = self.mo_occ_full_reidx.copy()
        self.mo_occ_act[self.remaining_cols] = 0
        self.mo_occ_env = self.mo_occ_full_reidx.copy()
        self.mo_occ_env[self.act_cols] = 0

        ## environment indices for occupied orbitals
        self.env_idx_occ = self.mo_occ_env.nonzero()[0]


        self.dm_full = self.global_scf_obj.make_rdm1(mo_coeff=self.C_full_reidx,
                                                     mo_occ=self.mo_occ_full_reidx)

        
        self.dm_act = self.global_scf_obj.make_rdm1(mo_coeff=self.C_full_reidx,
                                                    mo_occ=self.mo_occ_act)
        
        self.dm_env = self.global_scf_obj.make_rdm1(mo_coeff=self.C_full_reidx,
                                                    mo_occ=self.mo_occ_env)


        self.E_nuclear    = self.global_scf_obj.energy_nuc()
        self.E_DFT_global = self.global_scf_obj.energy_elec(dm=self.dm_full)[0]
        self.E_env        = self.global_scf_obj.energy_elec(dm=self.dm_env)[0]
        self.E_act        = self.global_scf_obj.energy_elec(dm=self.dm_act)[0]
        self.E_cross      = self.E_DFT_global - self.E_env - self.E_act
                                     

        if self.SCF_type == "open-shell":
            ## need to deal with spin in this approach!
            veff_glob = self.global_scf_obj.get_veff(dm=self.dm_full) 
            G_glob = scf.rohf.get_roothaan_fock((veff_glob[0],veff_glob[1]), self.dm_full, self.Sao)

            veff_act = self.global_scf_obj.get_veff(dm=self.dm_act) 
            G_act = scf.rohf.get_roothaan_fock((veff_act[0],veff_act[1]), self.dm_act, self.Sao)
        else:
            G_glob = self.global_scf_obj.get_veff(dm=self.dm_full)
            G_act = self.global_scf_obj.get_veff(dm=self.dm_act)
     

        # embedding potential in AO basis (note common hcore_ao term cancels!)
        self.G_emb_ao = G_glob - G_act


        assert np.allclose(self.dm_full, self.dm_act + self.dm_env), "density matrices of act and env do not match full one"

        cpu_obj = global_scf_obj.to_cpu()
        nelec_active = (int((cpu_obj.mo_occ_act>0).sum()),
                        int((cpu_obj.mo_occ_act>1).sum())
                        )gi
        coords   = cpu_obj.mol.atom_coords(unit=cpu_obj.mol.unit)
        atm_list = [cpu_obj.mol.atom_pure_symbol(i) for i in range(cpu_obj.mol.natm)]
        self.mol_act = gto.M(
            atom=zip(atm_list, coords),
            unit=cpu_obj.mol.unit,
            basis=cpu_obj.mol.basis,
            charge=cpu_obj.mol.charge + cpu_obj.mol.nelectron - sum(nelec_active),
            spin=nelec_active[0] - nelec_active[1],
            max_memory=max_memory_MB,
        ).build()

        assert self.mol_act.nelectron == sum(nelec_active)

    def get_mu_projector(self):
        C_env_occ = self.C_full_reidx[:, self.env_idx_occ]
        P_env_ao = (self.Sao @ C_env_occ @ C_env_occ.T @ self.Sao)
        return P_env_ao

    def get_huz_projector(self):
        """Idempotent AO-basis projector onto the occupied environment: P = D_env S.

        P is idempotent but NOT symmetric, so it cannot be added to a Fock matrix
        as it stands. Use get_huz_operator for that.
        """
        C_env_occ = self.C_full_reidx[:, self.env_idx_occ]
        P_env_huz = C_env_occ @ C_env_occ.conj().T @ self.Sao
        return P_env_huz

    def get_huz_operator(self, Fao, level_shift:float=0):
        """The hermitian Huzinaga level shift -(F P + P^dag F) for a given Fock matrix.

        With P = D_env S, this annihilates the environment orbitals' own eigenvalues
        and returns them with the opposite sign, so the occupied environment block is
        pushed above the active HOMO and aufbau filling cannot reach it. Unlike the
        mu-shift there is no arbitrary constant: the shift is set by the environment
        orbital energies themselves.

        level_shift optionally adds a constant on top. Because the huzinaga term already
        removes the active-environment coupling *exactly*, a constant only relocates the
        already decoupled environment block and cannot reintroduce the mu-shift's 1/mu
        error. It buys aufbau headroom, which matters for ROKS: the roothaan effective
        fock averages focka and fockb, so a singly occupied environment orbital can have
        a positive eigenvalue, and the sign flip then moves it *down* rather than up.
        """
        FP = Fao @ self.get_huz_projector()
        O_huz = -(FP + FP.conj().T)
        if level_shift>0:
            O_huz = O_huz + level_shift * self.get_mu_projector()
        return O_huz

    def get_block_eigenvalues(self, cols, Fao=None):
        """Eigenvalues of the fock matrix restricted to a set of reindexed columns.

        Not the same as global_scf_obj.mo_energy: those are the canonical eigenvalues,
        whereas the huzinaga sign flip acts on whatever subspace the caller handed in,
        which is usually localised and so has a non-diagonal fock block. It is the
        eigenvalues of that block that get their sign flipped.
        """
        if Fao is None:
            Fao = self.global_scf_obj.get_fock(dm=self.dm_full)
        Fao = np.asarray(Fao)
        if Fao.ndim == 3:
            Fao = scf.rohf.get_roothaan_fock((Fao[0], Fao[1]), self.dm_full, self.Sao)
        C_sub = self.C_full_reidx[:, cols]
        return np.linalg.eigvalsh(C_sub.conj().T @ Fao @ C_sub)

    def huz_shift_threshold(self, Fao=None):
        """How large huz_level_shift must be for the environment to clear the fragment.

        Huzinaga returns environment eigenvalues negated, so the environment's lowest
        level lands at -max(eps_env). Filling stays correct only while that clears the
        fragment HOMO, giving lambda > max(eps_env) + eps_homo_frag. Both are taken
        from the global fock here, which is all that is known before the embedded SCF
        runs, so treat the number as an estimate rather than a guarantee.
        """
        eps_env = self.get_block_eigenvalues(self.env_idx_occ, Fao)
        eps_act = self.get_block_eigenvalues(self.mo_occ_act.nonzero()[0], Fao)
        return eps_env.max() + eps_act.max(), eps_env, eps_act.max()

    def warn_huz_positive_env(self, huz_level_shift:float=0):
        """Warn when the huzinaga sign flip would push an environment level downwards.

        A positive occupied-environment eigenvalue is reflected to a *negative* one, so
        instead of being lifted out of the way it can drop below the fragment HOMO and
        get filled. Returns True when a warning was issued.
        """
        threshold, eps_env, eps_homo = self.huz_shift_threshold()
        n_pos = int((eps_env > 0).sum())
        if n_pos == 0 or huz_level_shift > threshold:
            return False

        print(
            f"\n!!! HUZINAGA WARNING\n"
            f"    {n_pos} of {len(eps_env)} occupied environment orbitals have a POSITIVE\n"
            f"    fock eigenvalue (largest {eps_env.max():+.4f} Ha). The huzinaga sign flip\n"
            f"    sends that level DOWN to {-eps_env.max():+.4f} Ha instead of lifting it up,\n"
            f"    and the fragment HOMO is near {eps_homo:+.4f} Ha, so aufbau filling may put\n"
            f"    fragment electrons into the environment.\n"
            f"    Pass huz_level_shift >= {threshold + 0.5:.2f} (or simply something large like\n"
            f"    1e6 -- a constant shift costs no accuracy) instead of the current"
            f" {huz_level_shift:g}.\n"
            f"    Continuing anyway. Verify with check_embedding, and confirm that\n"
            f"    DFT-in-DFT reproduces the global DFT energy.\n"
        )
        return True

    def warn_not_converged(self, mf, proj_type=""):
        """Warn when the embedded SCF never converged, whatever the projector.

        Worth its own check because it is invisible to the orthogonality and aufbau
        tests: the orbitals can be perfectly clean and the energy still wrong by a long
        way. Returns True when a warning was issued.
        """
        if getattr(mf, "converged", True):
            return False

        extra = ("    A huz_level_shift also helps here: separating the fragment and\n"
                 "    environment blocks damps the oscillation.\n") if proj_type == "huz" else ""
        print(
            f"\n!!! WARNING: the embedded SCF did NOT converge\n"
            f"    E = {mf.e_tot:.8f} Ha is not trustworthy, and no orbital test will show\n"
            f"    it -- the orbitals can look perfectly clean. Raise max_cycle, try a\n"
            f"    different dm0, or damp the iterations.\n{extra}"
        )
        return True

    def warn_huz_aufbau_violated(self, mf, env_cols, tol=1e-8):
        """Warn when the converged embedding actually did fill an environment orbital.

        The companion to warn_huz_positive_env: that one predicts trouble from the global
        fock, this one detects it after the fact and is the definitive test. Returns True
        when a warning was issued.
        """
        occ = mf.mo_occ > 0
        if not len(env_cols) or not occ.any():
            return False

        C_env = self.C_full_reidx[:, self.env_idx_occ]
        overlap = np.abs(C_env.conj().T @ self.Sao @ mf.mo_coeff[:, occ]).max()
        margin = mf.mo_energy[env_cols].min() - mf.mo_energy[occ].max()
        if margin > 0 and overlap < tol:
            return False

        print(
            f"\n!!! HUZINAGA WARNING: the converged embedding is contaminated\n"
            f"    aufbau margin              {margin:+.4f} Ha  (must be > 0)\n"
            f"    max |<env occ|S|emb occ>|  {overlap:.2e}      (must be ~0)\n"
            f"    An environment orbital sits at or below the fragment HOMO, so the\n"
            f"    fragment has occupied orbitals that belong to the environment and the\n"
            f"    energy is not trustworthy. Increase huz_level_shift.\n"
        )
        return True

    def build_emb_dft(self, xc_expensive, proj_type="mu", dm0=None, huz_level_shift:float=0,
                      warn:bool=True, scf_modify_function=None):
        
        if self.SCF_type == "open-shell":
            dft_emb = dft.ROKS(self.mol_act, xc=xc_expensive)
        else:
            dft_emb = dft.RKS(self.mol_act, xc=xc_expensive)

        ## use same settings as global SCF
        dft_emb.verbose = self.global_scf_obj.verbose
        dft_emb.max_cycle = self.global_scf_obj.max_cycle
        dft_emb.conv_tol = self.global_scf_obj.conv_tol

        if scf_modify_function is not None:
            ### modifications done before embedding. This
            ### could be used to add an embedding to the SCF object, or to add a solvent, add GPU support (gpu4pyscf), etc.
            ### can also override global SCF settings above!
            dft_emb = scf_modify_function(dft_emb)

        ## get standard hcore in AO basis
        hcore_std = dft_emb.get_hcore()

        if proj_type == "mu":
            P_env_ao = self.get_mu_projector()
            v_emb = self.mu_val*P_env_ao + self.G_emb_ao
            hcore_mod = hcore_std + v_emb
            dft_emb.get_hcore = lambda *args, **kwargs: hcore_mod

        elif proj_type == "huz":
            ## the huzinaga shift depends on the running fock matrix, so it cannot live in
            ## hcore: PySCF's kernel evaluates get_hcore once before the SCF loop, which would
            ## freeze the shift at the initial guess. Override get_fock instead and leave
            ## hcore holding the density-independent embedding potential only, which also
            ## keeps energy_elec free of the projector.
            if warn:
                self.warn_huz_positive_env(huz_level_shift)

            hcore_mod = hcore_std + self.G_emb_ao
            dft_emb.get_hcore = lambda *args, **kwargs: hcore_mod

            get_fock_std = dft_emb.get_fock

            def get_fock_huz(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                             diis=None, **kwargs):
                
                if dm is None:
                    dm = dft_emb.make_rdm1()

                if h1e is None:
                    h1e = hcore_mod
                if vhf is None:
                    vhf = dft_emb.get_veff(dm=dm) # if dm is not None else dft_emb.get_veff()
                
                ## rebuild the shift from this cycle's fock, then let PySCF apply
                ## DIIS/damping/level-shift to the already-shifted matrix
                if len(vhf.shape) == 3:
                    focka = h1e + vhf[0]
                    fockb = h1e + vhf[1]
                    ## a restricted open-shell object can still be handed a spin-summed dm
                    ## (the initial guess is one), which get_roothaan_fock cannot unpack
                    ## 0.5 is needed for the case when dm is spin-summed (restricted setting!... aka split into, dm_a, dm_b)
                    dm_ab = dm if np.ndim(dm) == 3 else np.array((np.asarray(dm) * 0.5,) * 2)
                    Fao = scf.rohf.get_roothaan_fock((focka,fockb), dm_ab, self.Sao)
                else:
                    Fao = h1e + vhf

                h1e_huz = h1e + self.get_huz_operator(Fao, level_shift=huz_level_shift)
                
                ## return using the standard function! But with modified h1e!
                return get_fock_std(h1e=h1e_huz, s1e=s1e, vhf=vhf, dm=dm,
                                    cycle=cycle, diis=diis, **kwargs)

            dft_emb.get_fock = get_fock_huz
        else:
            raise ValueError(f"Invalid projection type: {proj_type}")
        
        ## start from the global DFT active density. The huzinaga shift only lifts the
        ## environment by |eps_env|, so from PySCF's minao guess the environment block of
        ## the fock matrix can come out positive, the sign flip then drives those orbitals
        ## *below* the active ones and aufbau locks onto the wrong (stable) state.
        if dm0 is None:
            dm0 = self.dm_act.copy()

        # if len(dm0.shape) == 3:
        #     dm0 = dm0 + dm0 ## add alpha and beta densities together

        dft_emb.kernel(dm0=dm0)
        ## modified - non-modified hcore
        v_emb_ao = dft_emb.get_hcore() - hcore_std

        # correction given with respect to original active density (not new one)
        if self.SCF_type == "open-shell":
            emb_corr = np.einsum("ij,ji->", self.dm_act[0], v_emb_ao) + np.einsum("ij,ji->", self.dm_act[1], v_emb_ao) 
        else:
            emb_corr = np.einsum("ij,ji->", self.dm_act, v_emb_ao) 
            
        
        # weight of every converged orbital on the occupied-environment space
        # useful to find env orbital in case solving changes the ordering!
        C_env = self.C_full_reidx[:, self.env_idx_occ]
        P_env = C_env @ C_env.conj().T
        weight = np.einsum("ji,jk,kl,li->i", dft_emb.mo_coeff, self.Sao, P_env, self.Sao @ dft_emb.mo_coeff)
        env_cols = np.where(weight > 0.5)[0]

        if warn:
            self.warn_not_converged(dft_emb, proj_type)
            if proj_type == "huz":
                self.warn_huz_aufbau_violated(dft_emb, env_cols)

        env_plus_corrections = self.E_env + self.E_cross - emb_corr
        E_dft_in_dft = dft_emb.e_tot + env_plus_corrections
        return E_dft_in_dft, dft_emb, emb_corr, env_cols, env_plus_corrections

    def build_emb_hf(self, proj_type="mu", dm0=None, huz_level_shift:float=0,
                     warn:bool=True, scf_modify_function=None):
        
        if self.SCF_type == "open-shell":
            hf_emb = scf.ROHF(self.mol_act)
        else:
            hf_emb = scf.RHF(self.mol_act)

        ## use same settings as global SCF
        hf_emb.verbose = self.global_scf_obj.verbose
        hf_emb.max_cycle = self.global_scf_obj.max_cycle
        hf_emb.conv_tol = self.global_scf_obj.conv_tol

        if scf_modify_function is not None:
            ### modifications done before embedding. This
            ### could be used to add an embedding to the SCF object, or to add a solvent, etc.
            ### can also override global SCF settings above!
            hf_emb = scf_modify_function(hf_emb)


        ## use same settings as global SCF
        hf_emb.verbose = self.global_scf_obj.verbose
        hf_emb.max_cycle = self.global_scf_obj.max_cycle
        hf_emb.conv_tol = self.global_scf_obj.conv_tol

        ## get standard hcore in AO basis
        hcore_std = hf_emb.get_hcore()

        if proj_type == "mu":
            P_env_ao = self.get_mu_projector()
            v_emb = self.mu_val*P_env_ao + self.G_emb_ao
            hcore_mod = hcore_std + v_emb
            hf_emb.get_hcore = lambda *args, **kwargs: hcore_mod

        elif proj_type == "huz":
            ## the huzinaga shift depends on the running fock matrix, so it cannot live in
            ## hcore: PySCF's kernel evaluates get_hcore once before the SCF loop, which would
            ## freeze the shift at the initial guess. Override get_fock instead and leave
            ## hcore holding the density-independent embedding potential only, which also
            ## keeps energy_elec free of the projector.
            if warn:
                self.warn_huz_positive_env(huz_level_shift)

            hcore_mod = hcore_std + self.G_emb_ao
            hf_emb.get_hcore = lambda *args, **kwargs: hcore_mod

            get_fock_std = hf_emb.get_fock

            def get_fock_huz(h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                             diis=None, **kwargs):
                
                if dm is None:
                    dm = hf_emb.make_rdm1()

                if h1e is None:
                    h1e = hcore_mod
                if vhf is None:
                    vhf = hf_emb.get_veff(dm=dm) # if dm is not None else dft_emb.get_veff()
                
                ## rebuild the shift from this cycle's fock, then let PySCF apply
                ## DIIS/damping/level-shift to the already-shifted matrix
                if len(vhf.shape) == 3:
                    focka = h1e + vhf[0]
                    fockb = h1e + vhf[1]
                    ## a restricted open-shell object can still be handed a spin-summed dm
                    ## (the initial guess is one), which get_roothaan_fock cannot unpack
                    dm_ab = dm if np.ndim(dm) == 3 else np.array((np.asarray(dm) * 0.5,) * 2)
                    Fao = scf.rohf.get_roothaan_fock((focka,fockb), dm_ab, self.Sao)
                else:
                    Fao = h1e + vhf

                h1e_huz = h1e + self.get_huz_operator(Fao, level_shift=huz_level_shift)
                
                ## return using the standard function! But with modified h1e!
                return get_fock_std(h1e=h1e_huz, s1e=s1e, vhf=vhf, dm=dm,
                                    cycle=cycle, diis=diis, **kwargs)

            hf_emb.get_fock = get_fock_huz
        else:
            raise ValueError(f"Invalid projection type: {proj_type}")
        
        ## start from the global DFT active density. The huzinaga shift only lifts the
        ## environment by |eps_env|, so from PySCF's minao guess the environment block of
        ## the fock matrix can come out positive, the sign flip then drives those orbitals
        ## *below* the active ones and aufbau locks onto the wrong (stable) state.
        if dm0 is None:
            dm0 = self.dm_act.copy()

        # if len(dm0.shape) == 3:
        #     dm0 = dm0 + dm0 ## add alpha and beta densities together

        hf_emb.kernel(dm0=dm0)
        ## modified - non-modified hcore
        v_emb_ao = hf_emb.get_hcore() - hcore_std

        # correction given with respect to original active density (not new one)
        if self.SCF_type == "open-shell":
            emb_corr = np.einsum("ij,ji->", self.dm_act[0], v_emb_ao) + np.einsum("ij,ji->", self.dm_act[1], v_emb_ao) 
        else:
            emb_corr = np.einsum("ij,ji->", self.dm_act, v_emb_ao) 
            
        
        # weight of every converged orbital on the occupied-environment space
        # useful to find env orbital in case solving changes the ordering!
        C_env = self.C_full_reidx[:, self.env_idx_occ]
        P_env = C_env @ C_env.conj().T
        weight = np.einsum("ji,jk,kl,li->i", hf_emb.mo_coeff, self.Sao, P_env, self.Sao @ hf_emb.mo_coeff)
        env_cols = np.where(weight > 0.5)[0]

        if warn:
            self.warn_not_converged(hf_emb, proj_type)
            if proj_type == "huz":
                self.warn_huz_aufbau_violated(hf_emb, env_cols)

        env_plus_corrections = self.E_env + self.E_cross - emb_corr
        E_hf_in_dft = hf_emb.e_tot +  env_plus_corrections
        return E_hf_in_dft, hf_emb, emb_corr, env_cols, env_plus_corrections

    def check_embedding(self, C_act_embedded, mo_occ_act_embedded, mo_energy_act_embedded, Sao, label):
        """Is the embedding sound? Select orbitals by *what they are*, never by column index.

        emb_obj.act_cols is a positional window into the pre-embedding ordering. The embedded
        SCF returns its own orbitals sorted by energy, so a column index carries no meaning
        here. With the mu-shift you get away with it because the environment is parked at
        +mu, i.e. always the last columns. Huzinaga shifts each environment orbital by only
        |eps_env|, so they land scattered among the active virtuals and act_cols then picks
        an environment orbital up, which looks like a broken projector but is not.
        """
        C_env = self.C_full_reidx[:, self.env_idx_occ]
        P_env = C_env @ C_env.conj().T

        # weight of every converged orbital on the occupied-environment space
        weight = np.einsum("ji,jk,kl,li->i", C_act_embedded, Sao, P_env, Sao @ C_act_embedded)
        env_cols = np.where(weight > 0.5)[0]
        occ_cols = np.where(mo_occ_act_embedded > 0)[0]

        # the WF-in-DFT requirement: occupied embedded orbitals span none of the environment
        ovlp_occ = C_env.conj().T @ Sao @ C_act_embedded[:, occ_cols]
        margin = mo_energy_act_embedded[env_cols].min() - mo_energy_act_embedded[occ_cols].max()

        print(f"--- {label} ---")
        print("  environment landed in columns :", env_cols)
        print("  eps(environment)              :", np.around(mo_energy_act_embedded[env_cols], 4))
        print("  occupied columns              :", occ_cols)
        print("  eps(occupied)                 :", np.around(mo_energy_act_embedded[occ_cols], 4))
        print(f"  max |<env occ| S |emb occ>|   : {np.abs(ovlp_occ).max():.2e}   <- must be ~0")
        print(f"  aufbau margin                 : {margin:+.4f} Ha  <- must be > 0")
        assert np.abs(ovlp_occ).max() < 1e-8, "occupied orbitals are contaminated by the environment"
        assert margin > 0, "an environment orbital sits below the active HOMO"

        return env_cols

    def get_mo_integrals(self, act_emb_scf_obj, act_emb_C, norb, nelecas, mo_cas_idxs=None):
        """
        Get spatial MO integrals for the embedded system.
        if mo_cas_idxs is None, then use all MOs in the active space
        """
        if mo_cas_idxs is None:
            mo_cas_idxs = np.arange(norb)

        assert len(mo_cas_idxs) == norb

        cpu_act_emb_scf_obj = act_emb_scf_obj.to_cpu()
        assert np.sum(nelecas)<= np.sum(act_emb_scf_obj.mol.nelec)
        cas_act_emb = mcscf.CASCI(
                                 cpu_act_emb_scf_obj,
                                  norb,
                                  nelecas,
                                #   ncore=self.ncore
                                  )
        ## need to overwrite CASCI hcore function
        cas_act_emb.get_hcore = lambda *args, **kwargs: cpu_act_emb_scf_obj.get_hcore()
        
        C_emb_ordered_subspace = mcscf.addons.sort_mo(cas_act_emb, act_emb_C, 
                                                      mo_cas_idxs, base=0)
        
        h1_emb_mo, energy_core_emb = cas_act_emb.get_h1eff(mo_coeff=C_emb_ordered_subspace)
        eri_cas_mo_S4 = cas_act_emb.get_h2eff(mo_coeff=C_emb_ordered_subspace)

        # ## move back to numpy arrays!
        # energy_core_emb = float(energy_core_emb)
        # h1_emb_mo = numpy.asarray(h1_emb_mo)
        # eri_cas_mo_S4 = numpy.asarray(eri_cas_mo_S4)

        return energy_core_emb, h1_emb_mo, eri_cas_mo_S4
