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
    - using mmseqs2 with 30% identity, end up with  
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
- realised i need to make holdout set so i can test against this - re ran clusternig and getting a kind of leakage from clusters that are subsets of other clusters. updating split.py before running the tests - keep the original easy search, then search_pairs (all vs all)
    - this comes from the way the clustering happens: using cascaded clustering, in which there are rounds where sensitivity is increased. after each round, each cluster is collapsed to its representative, to only representatives got to the next round for comparison. so say we have 3 chains, A, B, C, where A-B = 35% similar, B-C = 35% similar, and A-C = 20% similar, if B is picked as representative, then A, B, C are clustered together. But if A is picked, C doesn't amke that cluster. can remove cascade.
- i think a recurring thing here is due to the way i am doing the split, the test class will always contain the smaller protein families, as test is a smaller split. this means that there will probs be worse scores in test, bc the train set contains larger families that have been trained on. will probs need to do a holdout on the bigger set (in the to do list), but now that we have a pretty good compare set up we can just compare that.

- running into a hardware issue - one epoch takes >40s so kind of slow. can try a shorter max distace (maybe 4 angstroms rather tha 5), so looking over fewer bonds. also maybe add an early stopping - i.e. if no improvement after 15-20 epochs, stop there (similar to qsar)
- edges are important here so want somehting like a gineconv. will compare this and CGConv and GATv2 when i get to the experiment phase.

- starting off with patience = 10
- doing a run first to see how long it takes and get a baseline, then will run gcn with physics vs without to see if we even want to add that.

- ok another issue: 

- running against the (admittedly primitive) baselines, this took about 2 hours
- r2 measures how much better the model is than predicting the pkd mean each time (note, baseline scores can be negative bc they use train mean whereas r2 is test mean)
    - r2 = 1 - (rmse / val_std)^2 of that seed's val set
    - r2 > 0.75v means it's a better model than just predicting mean

- the run died - ran out of memory, updating so it saves incrementally so i don't lose everything when it stops

results
                   gcn_physics        gcn_plain
rmse            1.531 +- 0.091   1.612 +- 0.122
mae             1.203 +- 0.058   1.259 +- 0.090
pearson         0.408 +- 0.037   0.255 +- 0.142
spearman        0.410 +- 0.041   0.241 +- 0.135
r2              0.159 +- 0.037   0.067 +- 0.063

best_epoch       14.8 +-   6.3    20.2 +-  19.4
epochs_run       25.8 +-   6.3    31.2 +-  19.4
seconds         639   +- 153     739   +- 460

- the physics informed gcn is better on all 5 metrics, and also has a lower variance. it also reaches it's best epoch in fewer iterations and slighlty less time, although variance is larger for gcn (likely dues to seed 3 finishing on epoch 0)
- r2 score is pretty low: 0.159 for gcn physics, so not a whole let better than the mean (but, better than gcn plain at least)

 looking seed by seed:

         rmse                      mae                       r2                     pearson                   spearman                  best epoch / num epochs
seed    phys   plain    diff      phys   plain    diff      phys   plain    diff    phys   plain    diff      phys   plain    diff      phys      plain
   0   1.482   1.536  -0.054     1.182   1.209  -0.027     0.096   0.028  +0.068   0.344   0.236  +0.108     0.342   0.216  +0.126     23/34     13/24
   1   1.520   1.541  -0.021     1.235   1.218  +0.017     0.188   0.165  +0.023   0.436   0.419  +0.017     0.435   0.395  +0.040     10/21     50/61
   2   1.664   1.757  -0.093     1.212   1.299  -0.087     0.162   0.066  +0.096   0.412   0.299  +0.113     0.421   0.277  +0.144     17/28     28/39
   3   1.564   1.731  -0.167     1.271   1.396  -0.125     0.183  -0.001  +0.184   0.430   0.030  +0.400     0.445   0.029  +0.417     17/28      0/11
   4   1.423   1.495  -0.072     1.117   1.171  -0.054     0.163   0.076  +0.087   0.417   0.291  +0.126     0.408   0.289  +0.119      7/18     10/21

- pearson is showing correlation: not super high, but better for physics (also only shows correlation, not if the pkd values are close, hence need other metrics for calibration)
- spearman: shows the rank. again not super high, and note that this is across all proteins, not looking at rank within protein family (which might be interesting)

looking at baselines:
seed  n_val  val_std  baseline_mean  physics pred_std  plain pred_std
   0    697    1.559          1.600              0.753           0.531
   1    704    1.686          1.730              0.676           0.856
   2    693    1.819          1.935              0.715           0.717
   3    701    1.730          1.750              0.768           0.040
   4    698    1.556          1.611              0.553           0.586

and then looking at the span of pkd - i.e. how wide a variation of pkd the data covers
seed  y_true span  physics span  plain span
   0        10.20          4.25        3.42
   1         8.66          4.39        5.91
   2        13.64          4.75        4.19
   3        11.37          4.78        0.24
   4         9.34          3.65        3.12

can see that both the gcns predict within a really narrow span. seed 3 for the plain gcn is really narrow, and best epoch was 0, so it didnt improve upon that in the patience. already saw above that physics gets better in fewer epochs.


- Seed 0 — physics's weakest seed by correlation (pearson 0.344, r2 0.096) despite an easy val draw (std 1.559). It also ran longest to get there, best epoch 23. Both models underperform their other seeds here, so it looks like a hard train draw rather than a hard val one.
- Seed 1 — the one seed where the two are near-equivalent: diff −0.021 rmse, +0.017 pearson, and plain actually wins mae by 0.017. Plain got there by training 61 epochs to physics's 21, peaking at epoch 50 (its longest run, 1432 s vs 530 s). So on this draw plain can match physics, but needs ~2.7x the compute.
- Seed 2 — the hardest val draw (std 1.819, baseline_mean rmse 1.935). Both models' absolute rmse is worst here, but their relative standing is normal. This is the seed that most inflates the sd columns.
- Seed 3 — plain collapsed. Prediction span 0.24 pKd (5.46–5.70), pred_std 0.040, best val loss at epoch 0, stopped after 11 epochs. Early stopping restored the epoch-0 weights, so that result is an essentially untrained network emitting the mean — hence r2 −0.001 and pearson 0.030, both indistinguishable from baseline_mean. Physics on the same split is entirely normal (0.430 pearson, best epoch 17). Every "biggest gap" in the diff columns is this one failure.
- Seed 4 — the easiest draw and the best absolute numbers for both (physics rmse 1.423, r2 0.163). Physics converged fastest here, best epoch 7.


- looking at how different pkd bands are predicted:
                 gcn_physics              gcn_plain
predicted band   n     gap    rmse        n     gap    rmse
(0.0, 4.5]      164  +0.085   1.446      181  -0.295   1.609
(4.5, 5.0]      642  +0.179   1.574      476  -0.135   1.501
(5.0, 5.5]     1031  -0.085   1.479      879  -0.106   1.488
(5.5, 6.0]      904  -0.102   1.544     1353  -0.007   1.624
(6.0, 6.5]      410  -0.041   1.590      391  +0.200   1.800
(6.5, 7.0]      206  +0.127   1.444      159  +0.281   1.952
(7.0, 7.5]       91  +0.259   1.479       41  +0.472   2.008
(7.5, 20.0]      45  +1.175   2.085       13  +0.232   1.411
(gap = pred_mean − true_mean; positive means predicting higher pkd than true)

- looks like pkd over about 7 is worse: the pred mean is further away. admittedly, there are fewer examples in this band, but we are looking for higher pkd - i.e. complexes that bind more strongly (if this were a model to predict ligand binding targets, for example)
- this happens for both the physics and plain one, so it's not solved by physics alone. it's hard to make a comparison here, as n for both is small and n for gcn plain is really small. But physics seems a little better calibrated than the plain one, but need to investigate this further - it could just be chance that seed 3 of gcn plain was best at epoch 0, it doesn't necessarily prove that physics is better.
- going to update so that we save the train preds, rn only val


testing if the physics brnach actually does anything:

- the problem with "physics beats plain" is that the branch bundles 3 separate things, so the result doesn't say which one is affecting results:
    1. the 1/r + 1/r^6 basis functions
    2. the coefs w1, w2 being free to depend on r (outputs of previous layers which depend on r)
    3. the fact that it SUMS over interface edges only, while the main path is a mean pool over all nodes. a sum is extensive (more contacts = bigger number), a mean is size blind. binding energy is extensive and pkd correlates with ligand size, so this alone could be worth something
- and a 4th problem: gcn_physics has 52228 params vs gcn_plain's 39745, so it might just be that the extra 12483 params help

- leak: edge_attr is (inv_r, inv_r6, log_inv_r6) - ALL of it is distance. and the pair mlp input includes edge_proj(edge_attr). so w1 is literally computed from r:
    r -> (1/r, 1/r^6, log) -> Linear(3->64) -> part of the 192-dim pair input -> Linear(192->64) -> relu -> Linear(64->2) -> (w1, w2)
- so w1 = w1(r), and contribution = w1(r)*(1/r) + w2(r)*(1/r^6). set w1(r) = r*g(r) and you get contribution = g(r) i.e. ANY function of r. the 1/r form constrains nothing. it's only a lennard-jones law if w1, w2 are CONSTANTS - like epsilon and sigma in a real force field, which depend on which elements are involved but never on the current distance
- same as saying y = m*x is a straight line. only true if m is constant. if m = m(x) it can be any curve

tests:
- gcn_physics_blind: pair mlp gets the edge_type embedding only - a lookup by edge class, so the same 64 numbers for every lig-prot edge no matter the distance. w1, w2 become per-atom-pair-type constants = actual lennard-jones. verified by scaling inv_r/inv_r6 by 2: blind energy changes by exactly 2.000000, full physics by 2.068370. so w is reading r
- gcn_contact_sum: blind AND terms replaced by 1s, so energy = 0.5*sum(w1+w2) over interface edges. distance only enters through whether the edge exists at all (4A cutoff) - a 2.5A and a 4.0A contact between the same atom types contribute identically. basically contact counting weighted by what atoms are touching
    - it has to be blind too: if w can read r then w*1 and w*(1/r) span the same functions, so an unblinded contact_sum would just be gcn_physics under another name
- gcn_plain_wide: plain with hidden 75 instead of 64 -> 52351 params vs 52228,  to see if it's just extra params that improve performance
- the 3 physics tests have same param count so can compare them between each other

(collapse means jsut uses epoch 0 as best epoch)
results, 5 seeds, val:
                    params            rmse          pearson           r2     collapses
gcn_plain            39745   1.612 +- 0.122   0.255 +- 0.142   0.067          1/5
gcn_plain_wide       52351   1.622 +- 0.145   0.260 +- 0.142   0.056          1/5
gcn_contact_sum      52228   1.547 +- 0.094   0.379 +- 0.059   0.141          0/5
gcn_physics_blind    52228   1.531 +- 0.087   0.403 +- 0.040   0.158          0/5
gcn_physics          52228   1.531 +- 0.091   0.408 +- 0.037   0.159          0/5

what each test is worth
capacity alone     plain -> plain_wide       +0.0098  t=+0.30  1/5   p 0.78    nothing
interface sum      plain -> contact_sum      -0.0653  t=-3.22  5/5   p 0.032   ~80% of it
1/r, 1/r^6 basis   contact_sum -> blind      -0.0160  t=-1.67  3/5   p 0.17    ~20%
free distance      blind -> physics          -0.0002  t=-0.04  2/5   p 0.97    nothing

- capacity explains nothing. plain_wide is actually slightly worse than plain, so the effect is architectural. physics vs plain_wide is -0.0913, t=-3.03, p 0.039
- ~80% of the gain is just having an interface-restricted summed readout head. no distance law needed at all
- the 1/r + 1/r^6 basis adds ~20% but only 3/5 seeds, t=-1.67, p 0.17. doesn't separate at n=5 and would need ~15 seeds to resolve
- letting the coefs read r adds NOTHING (-0.0002, blind wins 3/5). so the constrained version means what it says and costs nothing - keep it. the branch was never cheating via the leak, it just isn't the physics doing the lifting
- collapses: 2/10 runs without the branch collapsed to predicting the mean (best epoch 0), 0/15 with it. and on DIFFERENT seeds (plain on 3, plain_wide on 2) so probs optimiser instability, not a bad split. fisher exact p ~ 0.14 so suggestive, not shown

- stats caveat - the t test here is a paired t on the 5 per-seed rmse diffs (same split for both models each seed, so the val draw cancels). df = 4, 95% crit 2.776. 
    - everything that clears 0.05 is marginal: p = 0.029, 0.032, 0.039, 0.042, 0.048. nothing comfortable
    - "5/5 seeds" on its own is NOT significant
    - and all of this is val, which also picks the early stopping epoch, so absolute numbers are optimistic. same bias for every test so the comparisons are still fair

so just adding the contacts is what's pushing it up

- so far ... physics is doing nothing :)

- and we are massively underfitting


to do
- run holdout test  Run a large-family holdout as a separate experiment: train on everything except the 532-cluster, test on that cluster alone.
- swap out layernorm for mean agg, maybe include degree count
- directional geometry - h-bonds are angle dependent and we only have distances
