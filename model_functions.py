############ import libraries ############

import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    
install('hdbscan')
install('py3Dmol')

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import hdbscan
import py3Dmol
import string
from sklearn.decomposition import NMF
from scipy.optimize import curve_fit
from random import choices
from scipy import stats
from scipy.special import comb
import networkx as nx
import matplotlib


############ helper functions ############

def get_path(PDB_name, appendix):
  seq = f"{PDB_name}.{appendix}"
  return seq

def parse_aln(filename):
  '''
  Returns: aligned sequences as a 2D array
  '''
  sequence = []
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    sequence.append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return np.array(sequence)

def get_gap_dict(gene_name):  
  '''
  gap_dict = dictionary from pdb to mtx_ref (deletes gaps)
  gap_dict_r = dictionary from mtx_ref to pdb (re-inserts gaps) 
  '''
  ref_file = open(get_path(gene_name,"mtx_ref"),'r').readlines() # returns 1D list
  ori_seq = ref_file[0].strip() # obtain original seq
  ref_seq = ref_file[1].strip()

  # calculate the length of sequence been used
  gap_dict = {}
  gap_dict_r = {}
  ncol = 0
  for i,AA in enumerate(ref_seq):
    if AA != '-':
      gap_dict[i] = ncol 
      gap_dict_r[ncol] = i
      ncol = ncol + 1     
  return gap_dict, gap_dict_r

# dict to convert amino acids from 1-letter abbrev to 3-letter
dict_aa = {'A':'ALA',
          'G':'GLY',
          'V':'VAL',
          'L':'LEU',
          'I':'ILE',
          'F':'PHE',
          'W':'TRP',
          'S':'SER',
          'Y':'TYR',
          'T':'THR',
          'N':'ASN',
          'Q':'GLN',
          'D':'ASP',
          'E':'GLU',
          'P':'PRO',
          'C':'CYS',
          'M':'MET',
          'K':'LYS',
          'R':'ARG',
          'H':'HIS',
          'B':'ASX'}


######### GREMLIN ##########

# importing functions specific to tensorflow v1
import tensorflow.compat.v1 as tf

# disable eager execuation (if using tensorflow v2)
tf.disable_eager_execution()

def parse_fasta(filename, a3m=False):
  '''function to parse fasta file'''
  
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      header.append(line[1:])
      sequence.append([])
    else:
      if a3m: line = line.translate(rm_lc)
      else: line = line.upper()
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  return header, sequence
  
def mk_msa(seqs):
  '''one hot encode msa'''
  alphabet = "ARNDCQEGHILKMFPSTWYV-"
  states = len(alphabet)  
  a2n = {a:n for n,a in enumerate(alphabet)}
  msa_ori = np.array([[a2n.get(aa, states-1) for aa in seq] for seq in seqs])
  return np.eye(states)[msa_ori]

from scipy.spatial.distance import pdist, squareform
def get_eff(msa, eff_cutoff=0.8):
  if msa.ndim == 3: msa = msa.argmax(-1)    
  # pairwise identity  
  msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
  # weight for each sequence
  msa_w = (msa_sm >= eff_cutoff).astype(float)
  msa_w = 1/np.sum(msa_w,-1)
  return msa_w

def GREMLIN_simple(msa, msa_weights=None, opt_iter=100, b_ini=None, w_ini=None):
  
  # collecting some information about input msa
  N = msa.shape[0] # number of sequences
  L = msa.shape[1] # length of sequence
  A = msa.shape[2] # number of states (or categories)
  
  # weights
  if msa_weights is None:
    msa_weights = np.ones(N)

  # kill any existing tensorflow graph
  tf.reset_default_graph()

  # setting up weights
  b = tf.get_variable("b", [L,A])
  w_ = tf.get_variable("w", [L,A,L,A], initializer=tf.initializers.zeros)

  # symmetrize w
  w = w_ * np.reshape(1-np.eye(L),(L,1,L,1))
  w = w + tf.transpose(w,[2,3,0,1])

  # input
  MSA = tf.constant(msa,dtype=tf.float32)
  MSA_weights = tf.constant(msa_weights,dtype=tf.float32)

  # dense layer + softmax activation
  MSA_pred = tf.nn.softmax(tf.tensordot(MSA,w,2)+b,-1)

  # loss = categorical crossentropy (aka pseudo-likelihood)
  loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(MSA,MSA_pred),-1)
  loss = tf.reduce_sum(loss * MSA_weights)

  # add L2 regularization
  reg_b = 0.01 * tf.reduce_sum(tf.square(b))
  reg_w = 0.01 * tf.reduce_sum(tf.square(w)) * 0.5 * (L-1) * (A-1)
  loss = loss + reg_b + reg_w

  # setup optimizer
  learning_rate = 0.1 * np.log(N)/L
  opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  # optimize!
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if b_ini is None:
      # initialize bias
      pc = 0.01 * np.log(N)
      b_ini = np.log(np.sum(msa,0) + pc)
      b_ini = b_ini - np.mean(b_ini,-1,keepdims=True)
    sess.run(b.assign(b_ini))

    if w_ini is not None:
      sess.run(w_.assign(w_ini))
    
    print(0, sess.run(loss))

    for i in range(opt_iter):
      sess.run(opt)
      if (i+1) % int(opt_iter/2) == 0:
        print((i+1),sess.run(loss))

    # save the weights (aka V and W parameters of the MRF)
    V = sess.run(b)
    W = sess.run(w)
  return(V,W)


def get_mtx(W):
  # l2norm of 20x20 matrices (note: we ignore gaps)
  raw = np.sqrt(np.sum(np.square(W[:,:-1,:,:-1]),(1,3)))

  # apc (average product correction)
  ap = np.sum(raw,0,keepdims=True)*np.sum(raw,1,keepdims=True)/np.sum(raw)
  apc = raw - ap
  np.fill_diagonal(apc,0)
  
  return raw, apc 


######## protein sector extraction #######

def get_binary_mtx(c, sig_pos, size):
  mat = c[None,:] == c[:,None]
  binary_mtx = np.zeros((size, size))
  for i in range(len(sig_pos)):
    for j in range(len(sig_pos)):
      binary_mtx[sig_pos[i], sig_pos[j]] = mat[i, j]
      binary_mtx[sig_pos[j], sig_pos[i]] = mat[j, i]
  return binary_mtx

def init_param(gene_name, opt_iter=1000):
  '''
  Extract bias V and weights W from GREMLIN using non-bootstrapped MSA 
  '''
  seqs = parse_aln(get_path(gene_name, 'aln'))
  msa = mk_msa(seqs)
  msa_weights = get_eff(msa)
  V,W = GREMLIN_simple(msa, msa_weights, opt_iter)
  raw, apc = get_mtx(W)
  return V, W

def getMaxCorr(x, y):
  '''Calculates the correlation between each row in x and each row in y, and
  returns the max correlation
  x, y: 2D arrays'''

  x_n = x.shape[0]
  mtx = np.zeros((x_n, x_n))
  for i in range(x_n):
    for j in range(x_n):
      _, _, r, _, _ = stats.linregress(x[i], y[j])
      mtx[i,j] = r
  maxR = 0
  mtx1 = np.copy(mtx)
  for i in range(x_n):
    idx = np.unravel_index(np.argmax(mtx), mtx.shape)
    maxR += mtx[idx]
    mtx[idx[0],:] = np.min(mtx)
    mtx[:,idx[1]] = np.min(mtx)
  return maxR/x_n, mtx1

def run_spectral_clustering(gene_name, opt_iter_init=1000, opt_iter=100, R_thres=0.99):
  '''
  Runs spectral clustering model for a given protein family. 
  APC = list of AP-corrected coevolution matrices; generated via GREMLIN from bootstrapped MSA's
  BINARY_MTX = list of binary matrices; summarize clusters 
  '''
  
  #initialize
  APC = []; BINARY_MTX = []; H = []
  n_bootstrap = 10
  iter = 0 

  print("Initializing...")
  V, W = init_param(gene_name, opt_iter_init)

  for n in range(n_bootstrap):
    
    print("Iter", iter, "...")
    iter+=1

    apc = bootstrap(gene_name, V, W, opt_iter)
    APC.append(apc)
    
    # sector extraction
    df, w_sigs, sig_pos = sector_identif(apc)
    if len(sig_pos)==0: c = np.array([])
    else:
      c = cluster_sectors(df, w_sigs, sig_pos, cluster_size=2, samples=2)
      # get common matrix
      rm_these = np.where(c==-1)[0]
      c = np.delete(c, rm_these)
      sig_pos = np.delete(sig_pos, rm_these)
    
    binary = get_binary_mtx(c, sig_pos, df.shape[0])
    BINARY_MTX.append(binary)

  # stop if no sectors
  if np.sum(BINARY_MTX)==0: return APC, BINARY_MTX

  # get nmf components
  num_sectors = elbow(BINARY_MTX)
  for n in range(n_bootstrap):
    consensus = np.mean(BINARY_MTX[:n+1],0)
    nmf = NMF(num_sectors).fit(consensus)
    H.append(nmf.components_)

  R = 0
  while R < R_thres:

    print("Iter", iter, "...")
    iter+=1

    apc = bootstrap(gene_name, V, W, opt_iter)
    APC.append(apc)
    
    # sector extraction
    df, w_sigs, sig_pos = sector_identif(apc)
    if len(sig_pos)==0: c = np.array([])
    else:
      c = cluster_sectors(df, w_sigs, sig_pos, cluster_size=2, samples=2)
      # get common matrix
      rm_these = np.where(c==-1)[0]
      c = np.delete(c, rm_these)
      sig_pos = np.delete(sig_pos, rm_these)
    
    binary = get_binary_mtx(c, sig_pos, df.shape[0])
    BINARY_MTX.append(binary)

    # check r
    consensus = np.mean(BINARY_MTX,0)
    nmf = NMF(num_sectors).fit(consensus)
    R = np.inf
    for n in range(n_bootstrap):
      r, _ = getMaxCorr(H[-n-1], nmf.components_)
      R = min(R, r)
    H.append(nmf.components_)

    print("R", round(R, 3))

  return APC, BINARY_MTX

def shuf_diag(coev):
  x = np.copy(coev)
  for i in range(1,x.shape[0]):
    np.random.shuffle(x.diagonal(i))
  fill_diag = np.triu(x)
  np.fill_diagonal(fill_diag,0)
  x = fill_diag + np.transpose(fill_diag)
  return x

# monte carlo simulation
def eigs_MC(coev, eig_num = 0, iter = 1000):
  total = np.array([])
  coev_copy = np.copy(coev)
  for i in range(iter):
    coev_copy = shuf_diag(coev_copy)
    # eigendecomp of shuffled matrix
    v_shuf, w_shuf, _ = np.linalg.svd(coev_copy)
    if eig_num != 0: total = np.concatenate((total, v_shuf.flatten()))
    else: total = np.concatenate((total, w_shuf))
  return total


def fit_curve(data):

  hist, bin_edges = np.histogram(data, density=True)
  bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

  def gauss(x, *p):
      A, mu, sigma = p
      return A*np.exp(-(x-mu)**2/(2.*sigma**2))

  p0 = [1., 0., np.std(data)]
  coeff, _ = curve_fit(gauss, bin_centres, hist, p0=p0)
  
  return coeff[1], coeff[2]


def get_thresholds(mean, std, num_stds = 2):
    std = abs(std)
    lower_thres = mean - num_stds*std
    upper_thres = mean + num_stds*std
    return lower_thres, upper_thres

def extract_sigVals(data, mean, std, num_stds = 2):

  # Determine the threshold above which eigenvalues are deemed significant
  lower_thres, upper_thres = get_thresholds(mean, std)

  # extract the significant eigenvalues
  condition1 = data > upper_thres
  extract1 = np.extract(condition1, data)

  condition2 = data < lower_thres
  extract2 = np.extract(condition2, data)

  data_sigs = np.concatenate((extract1,extract2))
  return data_sigs

def sector_identif(coev, num_stds = 2, num_iter = 200):

  # eigendecomposition
  v, w, _ = np.linalg.svd(coev)
  w = w[::-1]
  v = v[:,::-1] 
  df = pd.DataFrame(v)
  df['lambda'] = w

  ### spectral cleaning ###
  # random simulations
  total_w = eigs_MC(coev, iter = num_iter)
  data = np.concatenate((total_w, -total_w), axis=None)
  mean, std = fit_curve(data)
  # remove noise
  w_sigs = extract_sigVals(w, mean, std, num_stds)

  sig_pos = np.array([])
  # get only significant data points
  num_lambda = w.shape[0]
  # random simulations
  total_eigs = eigs_MC(coev, 1, iter = num_iter)
  mean, std = fit_curve(total_eigs)
  # determine the threshold above which eigenvalues are deemed significant
  lower_thres, upper_thres = get_thresholds(mean, std, num_stds)
  
  for eig_num in range(w_sigs.shape[0]):
    eig_num += 1
    # mark sector col of df
    colnum = num_lambda - eig_num
    eig_elements = v[:,colnum]

    idx1 = np.where(eig_elements > upper_thres)[0]
    idx2 = np.where(eig_elements < lower_thres)[0]
    if len(idx1)+len(idx2) != 0:
      sig_pos = np.concatenate([sig_pos, idx1, idx2])
  # arrange sig points in matrix
  sig_pos = np.unique(sig_pos)
  sig_pos = sig_pos.astype(int)

  return  df, w_sigs, sig_pos

def cluster_sectors(df, w_sigs, sig_pos, cluster_size=2, samples=2):
    
  if len(sig_pos)==1:
    return np.array([-1])

  else:  
    w = df['lambda'].to_numpy()
    size = w.shape[0]

    data = df.values[sig_pos,:-1][:,size-w_sigs.shape[0]:size]
    data = data * w_sigs 

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size,
                                min_samples=samples, 
                                gen_min_span_tree=True, 
                                prediction_data=False)
    clusterer.fit(data)
    hdbscan.HDBSCAN(algorithm='best', allow_single_cluster=False, alpha=1.0,
          approx_min_span_tree=True, cluster_selection_epsilon=0.0,
          cluster_selection_method='eom', core_dist_n_jobs=4,
          gen_min_span_tree=False, leaf_size=40,
          match_reference_implementation=False, 
          metric='euclidean', min_cluster_size=cluster_size, min_samples=samples, p=None)
    
    return clusterer.labels_ 

def bootstrap(gene_name, V_ini, W_ini, opt_iter=100):
  '''
  Bootstrap sampling of MSA and subsequent estimation of AP-corrected coevolution mtx
  V_ini = bias initialization for GREMLIN
  W_ini = weights initialization for GREMLIN
  opt_iter = number of iterations to run GREMLIN
  '''
  # bootstrap MSA
  seqs = parse_aln(get_path(gene_name, 'aln'))
  sample_seqs = choices(seqs, k=seqs.shape[0])
  msa = mk_msa(sample_seqs)

  # get APC
  msa_weights = get_eff(msa)
  V,W = GREMLIN_simple(msa, msa_weights, opt_iter, V_ini, W_ini)
  _, apc = get_mtx(W)
  
  return apc

def nmf_softmax(binary_mtx, components=4, lr=1e-2, alpha=0.0, l1_ratio=0.0):
  L = binary_mtx.shape[0]
  tf.reset_default_graph()

  nmf = NMF(components).fit(binary_mtx)
  W = nmf.fit_transform(binary_mtx)
  W_nobin = 1-np.sum(W, axis=1, keepdims=True)
  W = np.concatenate((W, W_nobin), axis=1)

  # initialize W and H
  alpha = tf.constant(alpha)
  l1_ratio = tf.constant(l1_ratio)
  W = tf.Variable((W-0.5)*10, dtype=tf.float32)

  W_prob = tf.nn.softmax(W,-1)

  binary_mtx_pred = W_prob[:,:-1] @ tf.transpose(W_prob[:,:-1])
  binary_mtx_true = tf.constant(binary_mtx,dtype=tf.float32)
  loss = (0.5*(tf.norm(binary_mtx-binary_mtx_pred, ord='fro', axis=(0,1))**2)
                     +(alpha*l1_ratio*tf.norm(W_prob[:,:-1], ord=1))
                     +(alpha*l1_ratio*tf.norm(tf.transpose(W_prob[:,:-1]), ord=1))
                     +(0.5*alpha*(1-l1_ratio)*(tf.norm(W_prob[:,:-1], ord='fro', axis=(0,1))**2))
                     +(0.5*alpha*(1-l1_ratio)*(tf.norm(tf.transpose(W_prob[:,:-1]), ord='fro', axis=(0,1))**2)))
  
  # optimize
  opt = tf.train.AdamOptimizer(lr)
      
  train_W = opt.minimize(loss, var_list=[W])
  train_W_init = train_W

  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(10):
      for n in range(500):
        # update W
        _,objval = sess.run([train_W, loss])

        # print progress
        if (n+1) % 500 == 0:
            print('iter %i, %f' % (n+1, objval))
  
      train_W = train_W_init

    for n in range(1000):
      # update W
      _,objval = sess.run([train_W, loss])

      # print progress
      if (n+1) % 500 == 0:
          print('iter %i, %f' % (n+1, objval))

    return sess.run(W_prob[:,:-1]), sess.run(tf.transpose(W_prob[:,:-1]))

def elbow(binary_mtx):
  '''
  Automate elbow selection for number of components in NMF 
  binary_mtx = list of binary matrices after multiple bootstrap runs
  num_sectors = optimal number of components/sectors
  '''
  SSE = []
  consensus = np.mean(binary_mtx,0)
  for n_comp in np.arange(1,11):
    nmf = NMF(n_comp).fit(consensus)
    W = nmf.fit_transform(consensus)
    H = nmf.components_
    sse = np.sum((consensus - (W @ H))**2)
    SSE.append(sse)
  low_loss = np.where(np.array(SSE) <= 0.1)
  if len(low_loss[0])>0:
    return low_loss[0][0]+1
  delta1 = -np.diff(SSE)
  delta2 = -np.diff(delta1)
  delta2 = np.insert(delta2, 0, 0)
  strength = delta2-delta1
  strength = np.append(strength, np.nan)
  strength[0] = np.nan
  num_sectors = np.nanargmax(strength)+1
  return num_sectors



def get_RS(gene_name, H, num_sectors):
  _, gap_dict_r = get_gap_dict(gene_name)

  resi = []; scale = []
  argmax = np.argmax(H, axis=0)

  for i in range(num_sectors):
    pos = np.where(argmax==i)[0]
    rp = []; rs = []
    for p in pos:
      s = H[i,p]
      if s > 0.01:
        p = gap_dict_r[p]
        rp.append(p)
        rs.append(s*2)
    resi.append(rp)
    scale.append(rs)
    
  return resi, scale

def get_sectors(gene_name, binary_mtx):
  print('Running NMF...')
  num_sectors = elbow(binary_mtx)
  consensus = np.mean(binary_mtx,0)
  W, H = nmf_softmax(consensus, num_sectors)
  resi, scale = get_RS(gene_name, H, num_sectors)
  return resi, scale


def view_sectors(gene_name, resi, scale): 
  p = py3Dmol.view()
  p.addModel(open(get_path(gene_name, 'pdb')).read())
  p.setStyle({'model':0},{'cartoon': {'color':'red'}})

  p.addModel(open(get_path(gene_name, 'pdb')).read())
  p.setStyle({'model':1},{'cartoon': {'color':'red'}})
  color = ['blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'white', 'pink', 'lightblue']

  for i in range(len(resi)):
    for j in range(len(resi[i])):
      p.setStyle({'model':1,'and':[{'resi':resi[i][j]}, {'atom':'CA'}]},{'sphere':{'color':color[i], 'scale':scale[i][j]}})

  p.zoomTo()
  p.show()
    
    
############# graph networks #################

def get_confind(gene_name):
    
    sequence = open(get_path(gene_name, 'mtx_ref'), "r").readlines()[0][:-1]
    seq_len = len(sequence)

    lines = open(get_path(gene_name, 'cf'), "r").readlines()
    binary_mat = np.zeros((seq_len, seq_len))

    for line in lines:
      if line[:7] == 'contact':
        line = line.split('\t')
        cf = float(line[3])
        if cf > 0.01:
          i = int(line[1].split(',')[1])
          j = int(line[2].split(',')[1])
          binary_mat[i,j] = 1
          binary_mat[j,i] = 1
    
    return binary_mat


def reinsert_gaps_in_apc(gene_name, binary_mat, apc):
  PDB_L, _ = binary_mat.shape
  L, _ = apc.shape

  ## reinsert gaps
  gap_dict, _ = get_gap_dict(gene_name)
  apc_reinsert = np.copy(apc)

  i=0
  for PDB_pos in gap_dict.keys():
    if PDB_pos!=i:
      apc_reinsert = np.insert(apc_reinsert, np.repeat(i,PDB_pos-i), 0, axis=1)
      apc_reinsert = np.insert(apc_reinsert, np.repeat(i,PDB_pos-i), 0, axis=0)
    i=PDB_pos+1

  j = apc_reinsert.shape[0]
  if j != PDB_L:
    apc_reinsert = np.insert(apc_reinsert, np.repeat(j,PDB_L-j), 0, axis=1)
    apc_reinsert = np.insert(apc_reinsert, np.repeat(j,PDB_L-j), 0, axis=0)
  
  return apc_reinsert


def plot_networks(Graphs, probs): 
  n_graph = len(Graphs)
  plt.figure(figsize=(5*n_graph, 5))

  for n in range(n_graph):
    G = Graphs[n]
    plt.subplot(1,n_graph,n+1)
    pos = nx.circular_layout(G)  
    labels = {k:k for k in pos}
    strong = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 3]  
    contact = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] == 2] 

    # nodes
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["gainsboro","royalblue"])
    nx.draw_networkx_nodes(G, pos, node_size = 1200, node_color = range(len(pos)), cmap=cmap)
    nx.draw_networkx_labels(G, pos, labels)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist = contact, width = 3, alpha = 0.5, edge_color = 'thistle')
    nx.draw_networkx_edges(G, pos, edgelist = strong, width = 3, edge_color = 'plum')

    ax = plt.gca()
    ax.margins(0.08)
    plt.title(round(probs[n], 3), fontsize=30)
    plt.axis("off")
    plt.tight_layout()
  
  plt.show()
    

def get_graphs_with_confind_edges(resi, confind_mtx):
  Graphs = [] 
  for i in range(len(resi)):
    Graphs.append(nx.Graph())

  probs = np.zeros(len(resi))
  for s in range(len(resi)):
    sector = resi[s]
    size = len(sector)
    for i in range(size):
      for j in range(i+1, size):
        Graphs[s].add_nodes_from([sector[i],sector[j]])
        if confind_mtx[sector[i], sector[j]]:
          probs[s]+=1
          Graphs[s].add_weighted_edges_from([(sector[i],sector[j],2)])
    probs[s]=probs[s]/size

  # heavier weight for largest connected component
  resi_ = []
  probs = np.zeros(len(resi))
  for s in range(len(resi)):
    node_set = []
    for node in resi[s]:
      nodes = nx.node_connected_component(Graphs[s], node)
      if len(nodes)>=len(node_set):
        sector = list(nodes)
        size = len(sector)
        prob = 0
        for i in range(size):
          for j in range(i+1, size):
            if confind_mtx[sector[i], sector[j]]:
              prob+=1
        prob=prob/len(resi[s])
        if prob>probs[s]:
          node_set = nodes
          probs[s]=prob
    resi_.append(list(node_set))

  for s in range(len(resi_)):
    sector = resi_[s]
    size = len(sector)
    for i in range(size):
      for j in range(i+1, size):
        Graphs[s].add_nodes_from([sector[i],sector[j]])
        if confind_mtx[sector[i], sector[j]]:
          Graphs[s].add_weighted_edges_from([(sector[i],sector[j],3)])

  return Graphs, probs


def get_graphs_with_coev_edges(resi, confind_mtx, apc_reinsert):
  Graphs = [] 
  for i in range(len(resi)):
    Graphs.append(nx.Graph())

  for s in range(len(resi)):
    sector = resi[s]
    size = len(sector)
    for i in range(size):
      for j in range(i+1, size):
        Graphs[s].add_nodes_from([sector[i],sector[j]])
  
  sort = np.argsort(np.triu(apc_reinsert).flatten())[::-1]
  seq_len = confind_mtx.shape[0]
  sort_i = sort//seq_len
  sort_j = sort%seq_len

  probs = np.zeros(len(resi))
  num_sectors = len(resi)
  num_to_extract = np.sum(np.triu(confind_mtx)>0.01)
  for k in range(num_to_extract):
    i = sort_i[k]
    j = sort_j[k]
    for s in range(num_sectors):
      sector = resi[s]
      size = len(sector)
      if (i in sector) and (j in sector):
        probs[s]+=1/size
        Graphs[s].add_weighted_edges_from([(i,j,2)])

  resi_ = []
  probs = np.zeros(len(resi))
  for s in range(len(resi)):
    node_set = []
    for node in resi[s]:
      nodes = nx.node_connected_component(Graphs[s], node)
      if len(nodes)>=len(node_set):
        sector = list(nodes)
        prob = 0
        for k in range(num_to_extract):
          i = sort_i[k]
          j = sort_j[k]
          if (i in sector) and (j in sector):
            prob+=1
        prob=prob/len(resi[s])
        if prob>probs[s]:
          node_set = nodes
          probs[s]=prob
    resi_.append(list(node_set))

  for s in range(len(resi_)):
    sector = resi_[s]
    size = len(sector)
    for k in range(num_to_extract):
      i = sort_i[k]
      j = sort_j[k]
      if (i in sector) and (j in sector):
        Graphs[s].add_weighted_edges_from([(i,j,3)])

  return Graphs, probs