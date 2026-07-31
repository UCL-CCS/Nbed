## build embedded SCF objects

import numpy as np
from pyscf import gto, scf, dft

class EmbedSCF():

    def __init__(self, global_scf_obj,
                 act_MO_idxs, env_MO_idxs, 
                  mo_coeff, mo_occ, Sao, max_memory_MB, mu_val=1e6):
        
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

        occ_act   = np.setdiff1d(occ_all, act_MO_idxs)
        vir_act   = np.setdiff1d(vir_all, act_MO_idxs)
        occ_env   = np.setdiff1d(occ_all, env_MO_idxs)
        vir_env   = np.setdiff1d(vir_all, env_MO_idxs)

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

        nelec_active = (int((self.mo_occ_act>0).sum()),
                        int((self.mo_occ_act>1).sum())
                        )

        coords   = self.global_scf_obj.mol.atom_coords(unit=self.global_scf_obj.mol.unit)
        atm_list = [self.global_scf_obj.mol.atom_pure_symbol(i) for i in range(global_scf_obj.mol.natm)]

        self.mol_act = gto.Mole(
            atom=zip(atm_list, coords),
            unit=self.global_scf_obj.mol.unit,
            basis=self.global_scf_obj.mol.basis,
            charge=self.global_scf_obj.mol.charge + self.global_scf_obj.mol.nelectron - sum(nelec_active),
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

    def get_huz_operator(self, Fao):
        """The hermitian Huzinaga level shift -(F P + P^dag F) for a given Fock matrix.

        With P = D_env S, this annihilates the environment orbitals' own eigenvalues
        and returns them with the opposite sign, so the occupied environment block is
        pushed above the active HOMO and aufbau filling cannot reach it. Unlike the
        mu-shift there is no arbitrary constant: the shift is set by the environment
        orbital energies themselves.
        """
        FP = Fao @ self.get_huz_projector()
        return -(FP + FP.conj().T)

    def build_emb_dft(self, xc_expensive, proj_type="mu", dm0=None):
        
        if self.SCF_type == "open-shell":
            dft_emb = dft.ROKS(self.mol_act, xc=xc_expensive)
        else:
            dft_emb = dft.RKS(self.mol_act, xc=xc_expensive)

        ## use same settings as global SCF
        dft_emb.verbose = self.global_scf_obj.verbose
        dft_emb.max_cycle = self.global_scf_obj.max_cycle
        dft_emb.conv_tol = self.global_scf_obj.conv_tol

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
                    Fao = scf.rohf.get_roothaan_fock((focka,fockb), dm, self.Sao)
                else:
                    Fao = h1e + vhf

                h1e_huz = h1e + self.get_huz_operator(Fao)
                
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
            dm0 = self.dm_act

        # if len(dm0.shape) == 3:
        #     dm0 = dm0 + dm0 ## add alpha and beta densities together

        dft_emb.kernel(dm0=dm0)
        ## modified - non-modified hcore
        v_emb = dft_emb.get_hcore() - hcore_std

        # correction given with respect to original active density (not new one)
        if self.SCF_type == "open-shell":
            emb_corr = np.einsum("ij,ji->", self.dm_act[0], v_emb) + np.einsum("ij,ji->", self.dm_act[1], v_emb) 
        else:
            emb_corr = np.einsum("ij,ji->", self.dm_act, v_emb) 
            
        
        E_dft_in_dft = dft_emb.e_tot +  self.E_env + self.E_cross - emb_corr
        return E_dft_in_dft, dft_emb, emb_corr


