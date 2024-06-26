
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
import warnings
from itertools import chain

def create_diffdock_ligand_features(ligand_path_sdf, ligand_path_mol2):
    ligand = read_molecule(ligand_path_sdf, ligand_path_mol2, sanitize=False, calc_charges=True, remove_hs=False)
    node_features = create_node_features(ligand)
    edge_features = create_edges_and_edge_features(ligand)
    return dict(chain(node_features.items(), edge_features.items()))


def create_node_features(ligand):
    """
    Create node features for the ligand graph.


    The ligand has HYDROGENS INCLUDED, contrary to the diffdock features.
    Parameters
    ----------
    ligand : rdkit.Chem.rdchem.Mol
        RDKit molecule object.

    Returns
    -------
    atom_features_list : list of list of 'features'
        List of atom features for each atom in the ligand.
    """
    ringinfo = ligand.GetRingInfo()
    atom_features_list = []
    for index, atom in enumerate(ligand.GetAtoms()):
        chiral_tag = str(atom.GetChiralTag())
        if chiral_tag in ["CHI_SQUAREPLANAR", "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL"]:
            chiral_tag = "CHI_OTHER"
        atom_features_list.append([safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(chiral_tag)),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(index)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(index, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(index, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(index, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(index, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(index, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(index, 8)),
        ])
        
    if ligand.GetNumConformers() == 0:
        # generate a conformer
        AllChem.EmbedMolecule(ligand)
    lig_coords = torch.from_numpy(ligand.GetConformer().GetPositions()).float()

    return {"diffdock_ligand_atom_features": torch.tensor(atom_features_list), 
            "diffdock_ligand_atom_coordinates":lig_coords}

def create_edges_and_edge_features(ligand):

    start_nodes, end_nodes, _edge_types = [], [], [] # start_nodes and end_nodes are the indices of the atoms at the endpoints of ligand bonds.
    for bond in ligand.GetBonds():
        start_nodes += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] # we want undirected edges, so we add both directions
        end_nodes += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        _edge_types += [BOND_TYPES[bond.GetBondType()]] if bond.GetBondType() not in {BT.UNSPECIFIED, BT.DATIVE}  else [0,0]
    
    edge_index = torch.tensor([start_nodes, end_nodes], dtype=torch.long)
    _edge_type = torch.tensor(_edge_types, dtype=torch.long)
    edge_attr = F.one_hot(_edge_type, num_classes=len(BOND_TYPES)) #4 possibilites of bond types by default. 

    return {"diffdock_ligand_edge_index": edge_index, 
            "diffdock_ligand_edge_attr": edge_attr}


#################### helper functions ####################

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
BOND_TYPES = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def _read_molecule(molecule_file, sanitize=True, calc_charges=True, remove_hs=False):
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError('Expect the format of the molecule_file to be '
                         'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)

    except Exception as e:
        # Print stacktrace
        import traceback
        msg = traceback.format_exc()
        print(f"Failed to process molecule: {molecule_file}\n{msg}")
        return None

    return mol

def read_molecule(molecule_file_sdf, molecule_file_mol2, sanitize=True, calc_charges=True, remove_hs=False):

    mol = _read_molecule(molecule_file_sdf, sanitize=sanitize, calc_charges=calc_charges, remove_hs=remove_hs)
    if mol is None:
        mol = _read_molecule(molecule_file_mol2, sanitize=sanitize, calc_charges=calc_charges, remove_hs=remove_hs)
        if mol is None:
            raise ValueError('Failed to read molecule from both .mol2 and .sdf files.')
        else:
            return mol
    else:
        return mol