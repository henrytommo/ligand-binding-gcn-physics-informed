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
- there are also some cases where there is a description text that contains Kd rather than the raw value, so for the simplest model will cutoff anything with len > 12 - this removed 7.
- i now have a very fragile index processing pipeline. now onto the actual protein and ligand files themselves. they're split into time ranges, and cross referenceing what's in them with the Kd index ones i have chosen: 1981-2000 (382 folders), 2001-2010 (1745), and 2011-2019 (4806). so, 6933 total. It would be nice to have split on maybe the first 2 sets of years, but should use more than that for the training split. first i will load in all the ligands/etc to see how many fail to get final counts
- before running the whole thing, going to split this preprocessing pipeline into separate modules. im thinking: index parsing, ligand parsing, protein pocket parsing, and physics features (more to be added later). 
- in the first year group, only fails one, where rdkit can't parse, that's fine.
- remove all Hs so consistent across all molecs
- make a separate label for metals

