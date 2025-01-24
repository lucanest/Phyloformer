import sys
sys.path.append("../pyrosetta/PyRosetta4.Release.python39.ubuntu.release-363/setup")

from pyrosetta import *
init()
from pyrosetta.toolbox import cleanATOM
from pyrosetta.teaching import *

from pyrosetta.rosetta.core.pack.task import *
from pyrosetta.rosetta.protocols import *
from pyrosetta.rosetta.core.select import *




# Charging the pdb file
# Let's get a pdb file: sperm-whale myoglobin: 4pnj   
pose = pose_from_pdb( "../data/pdb4pnj.ent" )

# let's look at the sequence: 
pose.sequence()

# It has many ZZZs at the end, let's clean:
cleanATOM("../data/pdb4pnj.ent")

# Read the clean version
pose_clean = pose_from_pdb("../data/pdb4pnj.clean.pdb")

# let's look at the sequence: 
pose_clean.sequence()


# Scoring
sfxn = get_score_function(True)

#print(sfxn)

def mutateSequence(pose, posi, amino, sfxn):

    # Select Mutate Position
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(mut_posi.apply(pose)))

    # Select Neighbor Position
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose)))

    # Select No Design Area
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(not_design.apply(pose)))

    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    # Disable design
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    # Enable design
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))
    
    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(sfxn)
    packer.task_factory(tf)

    #Perform The Move
    if not os.getenv("DEBUG"):
      packer.apply(pose)



print(pose_clean.sequence())
print("\t\tEnergy of the original sequence: " + str(sfxn(pose_clean)))

pose_copy = pose_clean.clone()
mutateSequence(pose_copy, 3, "L", sfxn)
print(pose_copy.sequence())
print("\t\tEnergy of the mutated sequence (back to WT): " + str(sfxn(pose_copy)))
mutateSequence(pose_copy, 3, "Y", sfxn)
print(pose_copy.sequence())
print("\t\tEnergy of the mutated sequence (deleterious change): " + str(sfxn(pose_copy)))
mutateSequence(pose_copy, 3, "S", sfxn)
print(pose_copy.sequence())
print("\t\tEnergy of the mutated sequence (milder change): " + str(sfxn(pose_copy)))

mutateSequence(pose_copy, 3, "L", sfxn)
print(pose_copy.sequence())
print("\t\tEnergy of the mutated sequence (back to WT): " + str(sfxn(pose_copy)))

print(pose_clean.sequence())
print("\t\tEnergy of the original sequence, again: " + str(sfxn(pose_clean)))

# These tests show that with the current mutateSequence function, when a mutation is performed, the backward mutation does not reproduce the original object.
