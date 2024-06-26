from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import py3Dmol

import warnings
from IPython.display import display

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
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


def show_molecule_3d(molecule_name, molecule_path_pattern = "/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{}/{}_ligand.mol2"):
    molecule_path = molecule_path_pattern.format(molecule_name, molecule_name)
    mol = read_molecule(molecule_path, sanitize=True, remove_hs=True)
    # Convert to MolBlock format
    mol_block = Chem.MolToMolBlock(mol)

    # Create a viewer object
    viewer = py3Dmol.view(width=800, height=600)

    # Add the molecule to the viewer
    viewer.addModel(mol_block, 'sdf')

    # Set visualization style
    viewer.setStyle({'stick': {}})

    # Enable rotation and zoom
    viewer.zoomTo()
    viewer.show()
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=(300, 300))
    display(img)
    return viewer, img