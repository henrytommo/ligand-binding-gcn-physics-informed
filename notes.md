# ligand binding affinity - gcn + physics informed nn

- following on from qsar-biodegradation, wanted to try some bio data and also a physics informed neural net. pdbbind+ dataset looks like a good place to start, predicting binding affinities.
- When reading about this dataset, saw some notes and papers on data leakage due to similar proteins. i started a project to try and build a mycelium nerual network, and while reading around that there were methods of splitting the data by chemical skeleton, to avoid the nn just memorising a backbone, so i imagine this isn't too far off.

the data
- downloaded PDBbind+ v2020.R1 dataset - just the index and the protein-ligand dataset as this is what i'll be looking at.
- it is some index files and then some data files split by year
- using biopandas for parsing the .pdb files and rdkit for .sdf. also using py3Dmol for visualisations jsut so i can look at stuff
- the biopandas labelling of atoms includes other info, e.g. the alpha carbon, H attached to alpha/beta carbon and the number. this is a throwback. but biopandas can remove all H if you remove these so don;t need to go through one by one
- the pdb files have 2 files that describe the protein: a _protein file and a _pocket file. the pocket file is just the atoms in the immediate vicinity of the binding site - probs way more useful for me (computationally), rather than gcn having to take in many thousands of atoms of a protein
- both mol and sdf provided, but will use sdf
- the index gives both Kd and pKd - probs best to use pKd so a smaller scale but let me read about this
    - yeah ok want pKd. also helps with thermodynamics eqns later on
- looking through the protein ligand binding index, there are 6835 with just Kd, 4901 with just Ki, and 107 with both. Looks like a common approach is to combine the two, but they do represent different things and comparing them, they are often different (sometimes even different orders of magnitude). Kd is without competition in the system, Ki is with competition. To start with, I'll look just at Kd (pKd) and go on from there. There are ~7K Kd, which, depending on the differet types of protein structures (i.e. similarities), may be just about enough. but i will find out as i go on.
- there are also some cases where there is a description text that contains Kd rather than the raw value, get rid of this
- i now have a very fragile index processing pipeline. now onto the actual protein and ligand files themselves. they're split into time ranges, and cross referencing what's in them with the Kd index ones i have chosen: 1981-2000 (382 folders), 2001-2010 (1745), and 2011-2019 (4806). so, 6933 total. It would be nice to have split on maybe the first 2 sets of years, but should use more than that for the training split. first i will load in all the ligands/etc to see how many fail to get final counts
- before running the whole thing, going to split this preprocessing pipeline into separate modules. im thinking: index parsing, ligand parsing, protein pocket parsing, and physics features (more to be added later). 
- in the first year group, only fails one, where rdkit can't parse, that's fine.
- remove all Hs so consistent across all molecs
- make a separate label for metals
- got caught out with CA as calcium rather than alpha carbon
- there are some outliers in the pkd values (one is fM, so much higher). consider removing.

# training splits and scaling
- similar proteins often adopt similar shapes. training the model over the whole dataset as is risks it just learning the structures of the proteins (i.e. overfitting), as mentioned above
- there is a program called MMSeqs2 that cluster similar proteins. it does this by looking strings of amino acids, and looking for similar substrings (k-mers) across different proteins. it has a lookup of similar enough k-mers and matchers on these.
    - uses both gapped and ungapped matching, i.e. shifting to make gaps if there are matching amino acids later on
- looking for above 30% similarity ("twilight zone") - below 25-30% similarity, structures are not similar. may be above 40% for long chains. so, want to cluster by about 30% similarity.
- same for ligands, but will focus on proteins for now otherwise the dataset will shrink a lot. 
- we started off with 6,934 complexes. 4,944 distinct proteins (same protein vs different ligands).
    - using mmseqs2 with 30% identity, end up with 1,374 clusters, of which the largest has 532 members. there are 636 clusters of one, and median cluster size is 2.
    - so the issue i am foreseeing is there will eb a load of clusters (and the entirety of a cluster needs to go in one of the split,s train/val/test, otherwise the model can learn from the structure and when a protein from the same family is in the test set it could just be learning from the previous structure). don't want to end up with all the biggest clusters in one split and the model ends up training on very similar things. so randomly put in each split until they make up roughly 80/10/10 or similar perc.
    - will put the biggest cluster in train, then randomly assign the others across the splits.
        - this means, then, that the biggest family will never get tested on. but we can try running an experiment on that family by training on others
- i think i can reuse some of the train/test/split mechanics from my qsar-biodegradation project. the reusable way that was set up was aimed at comparing different models (e.g. xgboost vs log reg vs gcn), but can try it here with different gcn experiments. also that was a classifcation project whereas this one is regression, so change that too.
- need to think about scaling too - there were loads of features in qsar biodeg, but to start with here some of our features cna be on a log scale. need to keep this in mind:
    - inv_r6 is over a 1.5–5 ang range, probs want to handle. could log it so linear in log r - but then the physics branch would need the raw (i.e. in the l-r potential). keep both maybe. or add log as a separate feature. idk will try it out. the mean is around -8.7, so offset this so it is zero.
    - atomic_num should be an embedding index (otherwise implies linear relationship)
    - vdw_radius is 1.5–2.1 Å, probs ok
    - y (pKd, 0.6–15.2, mean 5.9) think about htis - maybe make it centre around zero
- as the coords of these proteins are taken from diffraction patterns, i.e. from crystallised proteins, the atom coords are an average. for parts of the protein that can move, this means you end up with atoms able to be in multiple occupancies, with "altloc" in the pdb file showing the occupancy of an atom in the different coords. it only affects one pocket file, but can dedupe by keeping the first CA so no residues are repeated.
- using nn.embedding for element (i.e. get around using just the atomic num). also for node type, i.e. ligand/protein/cofactor/metal. also for edge type (e.g. prot-prot, lig-prot etc). will start with a residue type embedding (rather than atom type for now)

config.py
- the mapping of 3 letter residues to 1 letter: for some modified residues (e.g. CA), these are mapped to the parent (i.e. cysteine). this is so that mmseqs2 can do the similarity clustering. does mean that we lose some info, but we still have the atom mappings. e.g. dimethylarsenic cycteine is mapped to cysteine, but the elemnet embedding for arsenic is summed alongside the residue.

index.py
- kind = Kd for now, but this can be changed
- parses index files to give a dict of {pdb_code: pKd}. you can change kd to ki by passing a different kwarg to the func

ligand.py
- parses ligand sdf files
- remove Hs so that the ligand matches the pocket (heavy atoms only)
- reshape, but actually on testing for empty ligands PyG just accepts the empty array. leave in for consistency, but it's not really needed.
- returns coords (3 x n, row per atom), atomic_nums (n values), bonds (2 x b, 2 rows showing start and end of bond, b is number of bonds) and mol, which is the rdkit mol object (full molecular graph), which contains info about bond orders, formal charges, aromaticity, stereochem etc. but not using this yet, just keeping for now.

pocket.py
|  class   |   atoms   | share  |
|---------|---------|---------|
| protein  | 4,245,473 | 93.90% |
| ligand   | 271,862   | 6.01%  |
| cofactor | 2,392     | 0.05%  |
| metal    | 1,751     | 0.04%  |
- classifies atoms, doing protein last so anything that is e.g. both a metal and in the protein is classified as protein
- end up with coords, atomic nums, node types, residue types, and atoms
- pocket, so only a few hundred atoms rather than the whole protein

residues.py
- basically answering q: is this part of the protein chain?
- minimal unique identifier for a residue in a structure
- groups together the backbone (N, CA, C) per key (residue name)

features.py
- physics features
- node features are cached, but edge_attr is built when loaded. so adding node feats, have to rebuild
- deafult feat: vdv radius, from the vdw radii table
- keep atomic num, but don;t use for reasons stated above
- edge features:
    - inv_r = 1/r
    - inv_r6 = 1/r**6
    - distance = raw r
    - log_inv_r6 = log of 1/r6 so we don't get a huuge range. not used for physics branch though, as would mess up the l-j potential

graph.py
- make the parsed ligand and pocket into a PyG data object
- build_graph is cached, but add_edges happens when loaded (fairly quick - about 8s over 6900 graphs)
- build_graph:
    - ligand and pocket go into single graph
- pocket duplicates: for occasions where atoms are copied across both the ligand and protein file, so end up with duplicates: for example, when the ligand is a peptide, or for covlent inhibitors (bond to ligand), and metal ions in the pocket. not picked up by the dedupes earlier. need to remove these otherwise we're counting atoms in the same location twice
- returns PyG data object

build.py
- builds the dataset
- same order across builds so can compare
- error handling and logs to track what's going on

split.py
- pdbbind repeats the same target with different ligands, so a random split scores the model on targets it has already seen: 36.9% of a random test set shares a protein sequence with train. this file reduces it to about 6% (bc we use easy-cluster, not exhaustive)
- basically comparing the full proteins and looking for those that are more than 30% similar to cluster together. then splits complexes into train/val/test
- extract_sequences - again looking at backbones like in pocket.py. but over whole protein
- uses mmseqs2
- clusters chains, but splits the complexes in train/test/val, making sure that families are spread over the splits
- holdout cluster allows us to look at one family (i.e. the biggest, that's always in train)

spec.py
- similar to qsar-biodeg to help with experiments
- pKd is mean 5.87, std 1.75, encode/decode class for scaling. helps so when initialising it's nearer zero. fit to train, and would use these when running inference on test
- model spec allows us to test plain gnn vs physics informed one
- dims_of resizes if feats added
- evaluate: nothing from val or test used in fit, val is only used for early stopping, and test is only scored at the end

metrics.py
- include spearman ranking - this way, instead of just rmse, mse other regression metrics etc, can look at how well the model ranks the binding affinities in order. that way, we can see even if the model predicts poorly, if it ranks well then we can recalibrate etc.

baselines.py
- predict mean for everything and compare to this to see if it's ay better
- baseline_ligand_size: more atoms = more possible contacts, and energies add up. so if preds are near the ligand baseline, we are just counting atoms
- would be good to add other stuff as baseline, e.g. molecular weight, h bonds etc - can get lots of this from rdkit.

compare.py
- every model sees the same split on a given seed, so can compare between models
- edges aren't cached so built each time
- calcs metrics over the model over seeds
- holdout - this is where the largest cluster is the test
- limits and max epochs - this is going to take probs a while to train so can limit

gcn.py
- physics + gcn
- the edges have 1/r and 1/r ^6: representing electrostatics and london dispersion forces. have set cutoff at 4 angstroms, which might be too short to take into account all the long range electrostatic effects
- coefs learnt for the physics terms, but maybe this is too much felxibility - can try without to see what happens
- the coef for the energy number from physics starts at zero, so adds in only if useful



## training
- running into a hardware issue - one epoch takes >40s so kind of slow. can try a shorter max distace (maybe 4 angstroms rather tha 5), so looking over fewer bonds. also maybe add an early stopping - i.e. if no improvement after 15-20 epochs, stop there (similar to qsar)
- edges are important here so want somehting like a gineconv. will compare this and CGConv and GATv2 when i get to the experiment phase.


to do
- run holdout test  Run a large-family holdout as a separate experiment: train on everything except the 532-cluster, test on that cluster alone.
- swap out layernorm for mean agg, perhaps include degree count
- gcn_physics_blind with the distance removed from the pair input — and a matched-pair comparison. If performance is unchanged, keep the constrained version, because it means what it says.
