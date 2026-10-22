"""Common feature subtraction model of 'silent' mitral cells.

Self-contained NumPy implementation of the olfactory-bulb model in
Tootoonian, Kollo & Schaefer, "'Silent' mitral cells emerge from common
feature subtraction." No data files are required: the mixing matrix and
odours are generated synthetically, so every figure is fully reproducible
from a random seed.
"""
import sys
import numpy as np
from copy import copy

try:                       # tqdm only provides a progress bar; it is optional
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

class Computation:
    def __init__(self, *args, **kwargs):
        self.computed = False
        
    def compute(self, *args, **kwargs):
        raise NotImplementedError("compute() method not implemented")

def get_x_true(N,P):
    """ Generate P patterns of length N, where each pattern has a single non-zero entry,
    and the non-zero entries are evenly spaced across the patterns.
    """
    spacing = N/(P+1)
    return np.array([np.arange(N)==int(spacing*(i+1)) for i in range(P)])
           
class SilentMcs(Computation):
    def __init__(self, *args, seed = 0, M=50, N=200, n_odours = 12, k = 1,
                 t_init = 1.0,
                 t_whiff = 1.0,
                 first_trial = None,
                 n_trials = 200,
                 min_air_trials = 4,
                 max_air_trials = 6,
                 dt = 2e-3,
                 ga = 1.,
                 be = 3,
                 sd = 0.15,
                 al = 0.95,
                 eta = 0.2,
                 tau_la = 0.050,
                 tau_v = 0.025,
                 randomize_trials = True,
                 gated = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.seed = seed
        np.random.seed(seed)

        self.M = M
        self.N = N
        self.n_odours = n_odours
        self.k = k
        self.first_trial = first_trial if first_trial is not None else n_odours
        self.n_trials = n_trials

        self.t_init = t_init
        self.t_whiff = t_whiff
        self.T = t_whiff

        self.min_air_trials = min_air_trials
        self.max_air_trials = max_air_trials
 
        self.ga = ga
        self.be = be
        self.sd = sd
        self.al = al
        self.eta = eta
        self.tau_la = tau_la
        self.tau_v = tau_v
        self.gated = gated      # baseline update only during odour (False: at all times)

    def _setup_odours(self, randomize_trials = True):
        print("Setting up odours")
        nt_init = int(np.ceil(self.t_init//self.dt))
        nt_whiff= int(np.ceil(self.t_whiff//self.dt))

        self.activation_time = self.t_init + self.t_whiff * self.first_trial

        stim_inds = []
        which_pat = [-1]*nt_init
        for i in range(1, self.n_odours)[::-1]:
            stim_inds.append(len(which_pat))
            which_pat += [i]*nt_whiff
            which_pat += [-1]*2*nt_whiff
        which_pat +=[-1]*1*nt_whiff
        self.activation_index = len(which_pat)
        which_pat += [-1]*3*nt_whiff
        # Now start the random odour protocol
        curr_tr = 0
        while curr_tr < self.n_trials:
            if which_pat[-1] < 0: # Last trial was air
                stim_inds.append(len(which_pat))        
                which_pat += [(np.random.choice(self.n_odours-1) if randomize_trials else np.mod(curr_tr, self.n_odours-1)) + 1] * nt_whiff
                curr_tr   += 1
            else:
                n_air_trials = np.random.choice(range(self.min_air_trials, self.max_air_trials+1))
                which_pat += [-1] * nt_whiff * n_air_trials
                curr_tr   += n_air_trials

        self.which_pat = np.array(which_pat)
        self.t_full    = np.arange(len(which_pat))*self.dt
        self.nt        = len(self.t_full)
        self.stim_inds = stim_inds

        self.A = np.random.randn(self.M, self.N)
        self.A /= np.linalg.norm(self.A, axis=0)
        self.x_true = get_x_true(self.N, self.n_odours)
        self.ind_x_true = np.array([np.where(xi)[0] for xi in self.x_true])

        self.bg_odour = 0.9 * self.A @ self.x_true[0]
        self.odours   = {odour:(self.A @ self.x_true[odour] + self.bg_odour) for odour in range(1, self.n_odours)}

        self.o_bar = np.mean(np.array(which_pat) >= 0)
        print(f"Odours are set up, with o_bar = {self.o_bar:1.2f}")
        
        print("Done setting up odours")

    def _compute_responses(self, p_bl = [-1], t_min = None, t_bl_pre = 0.5):
        t_min = t_min or (self.t_full[0] + (self.t_full[-1]-self.t_full[0])/2)

        print(f"Computing baseline and odour responses using t_min = {t_min:1.2f} s, and t_bl_pre = {t_bl_pre:1.2f} s")

        od_resp_wnds = []
        od_tail_resp_wnds = []
        bl_resp_wnds = []
        which_pat = []

        curr_wnd = []
        first    = True

        nt, M = self.la.shape
        curr_pat = -1
        for i, (ti, p) in enumerate(zip(self.t_full, self.which_pat)):
            if ti < t_min:
                continue
            if p in p_bl: # t>t_min, and arrived at an empty block
                first = False
                if len(curr_wnd): # We've made it to the end of the current block
                    od_resp_wnds.append(curr_wnd) # Add the current block to the list of odour response windows
                    od_tail_resp_wnds.append(curr_wnd[len(curr_wnd)//2:])
                    # Add the baseline response window, which has length t_bl_pre, and ends at the start of the current block 
                    bl_resp_wnds.append(np.where((self.t_full>(self.t_full[curr_wnd[0]]-t_bl_pre)) & (self.t_full<self.t_full[curr_wnd[0]]))[0])
                    which_pat.append(curr_pat)
                    curr_wnd = []
            else: # An odour
                if not first:
                    curr_wnd.append(i) # Add the current index to the current block if we've already got the first bloc
                    curr_pat = p 

        od_resps = np.zeros((len(od_resp_wnds), M))
        bl_resps = np.zeros((len(od_resp_wnds), M))
        gc_tail_resps = np.zeros((len(od_resp_wnds), self.N))
        eps_in   = np.zeros((len(od_resp_wnds), self.k))

        # Compute the mean response in each odour and baseline window
        for i, (bw,ow,tw,p) in enumerate(zip(bl_resp_wnds,od_resp_wnds, od_tail_resp_wnds, which_pat)):
            od_resps[i,:] = np.mean(self.la[ow,:],axis=0)
            bl_resps[i,:] = np.mean(self.la[bw,:],axis=0)
            gc_tail_resps[i,:] = np.mean(self.x[tw,:], axis=0)
            ind_od = self.ind_x_true[p]
            eps_in[i, :] = self.x_true[p][ind_od] - gc_tail_resps[i, ind_od] 
    
    
        self.od_resps = od_resps
        self.bl_resps = bl_resps
        self.od_resp_wnds = od_resp_wnds
        self.bl_resp_wnds = bl_resp_wnds
        self.gc_tail_resps = gc_tail_resps
        self.od_which_pat = which_pat
        self.eps_in = eps_in
        print(f"Computed responses for {len(od_resp_wnds)} odour blocks and their corresponding baseline blocks.")

        print("Done computing baseline and odour responses.")

    def compute(self, *args, run_mode = "common_feature", t_min = None, randomize_trials = True, dt=2e-3):
        assert run_mode in ["common_feature", "background_subtraction"], f"Invalid run_mode: {run_mode}. Must be 'common_feature' or 'background_subtraction'." 

        sys.stdout.flush()

        self.dt = dt
        
        self._setup_odours(randomize_trials=randomize_trials)       
            # We'll have the average on-time now, o_bar, so compute the related parameters
        sd2    = self.sd**2
        al2    = self.al**2
        eta2   = self.eta**2
        #eta2_inf = (-b + np.sqrt(b**2 + 4 * al2 * eta2 * sd2))/2/al2
        # Eqn var_ss
        eta2_inf = - eta2/al2
        num = np.sqrt(4 * al2 * eta2 * sd2 * self.o_bar + (eta2 + sd2 * (1 - al2))**2)
        num += (al2 - 1)*sd2 + eta2
        den = 2 * al2 * (1 - al2*(1 - self.o_bar))
        eta2_inf += num/den
        
        self.sd2_eff = sd2 + al2 * eta2_inf + eta2
        self.tau_b    = -self.T/np.log(self.al)
        self.ka      = self.al * (eta2 + al2 * eta2_inf)/(1 - self.al)

        print(f"η2     = {eta2:1.3e}")
        print(f"η2_inf = {eta2_inf:1.6f}")
        print(f"σ2_eff = {self.sd2_eff:1.6f}")
        print(f"τ_b    = {self.tau_b:1.3f}")
        print(f"κ (λ)  = {self.ka:1.3f}")

    
        print("Running SilentMcs dynamics with run_mode =", run_mode)
        la = np.zeros((self.nt, self.M))
        b  = np.zeros((self.nt, self.M))
        v  = np.zeros((self.nt, self.N))
        y  = np.zeros((self.nt, self.M))
        y_air = np.random.randn(self.nt, self.M)/np.sqrt(self.M)/10


        for i, t in tqdm(enumerate(self.t_full[1:])):
            odour = self.which_pat[i+1]

            if run_mode == "common_feature":
                # Common feature is only present with the odour, background is air.
                # in_odour is only truen when actually in the odour
                y, in_odour = (y_air[i], False) if odour < 0 else (copy(self.odours[odour]), True)
            elif run_mode == "background_subtraction":
                # Background odour is bg_odour and is always present. In_odour is always true
                y, in_odour = (copy(self.bg_odour), True) if odour < 0 else (copy(self.odours[odour]), True)
            else:
                raise ValueError(f"Invalid run_mode: {run_mode}. Must be 'common_feature' or 'background_subtraction'.")
            
            x = (v[i] - self.be) * (v[i] > self.be) / self.ga

            dladt = (-la[i] + (y - self.A @ x + b[i]) / self.sd2_eff) / self.tau_la
            dvdt  = (-v[i] + self.A.T @ la[i]) / self.tau_v
            gate  = in_odour if self.gated else 1.0
            dbdt  = (-b[i] - gate * self.ka * la[i]) / self.tau_b

            la[i+1] = la[i] + dladt * self.dt
            b[i+1]  = b[i]  + (dbdt  * self.dt) * (i > self.activation_index)
            v[i+1]  = v[i]  + dvdt  * self.dt

        self.x = (v - self.be) * (v > self.be) / self.ga
        self.y = y
        self.la = la
        self.b  = b
        self.v  = v

        print("Done computing SilentMcs dynamics.")
        self._compute_responses(t_min = t_min)
        
        self.computed = True

        return self
