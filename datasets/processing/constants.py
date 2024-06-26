import rdkit.Chem as Chem


LETTER_TO_NUM_TANKBIND = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "E": 5,
    "Q": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}

THREE_TO_NUMBER_DIFFDOCK = {'ALA': 0,
                            'ARG': 1,
                            'ASN': 2,
                            'ASP': 3,
                            'CYS': 4,
                            'GLU': 5,
                            'GLN': 6,
                            'GLY': 7,
                            'HIS': 8, 
                            'ILE': 9, 
                            'LEU': 10,
                            'LYS': 11,
                            'MET': 12, 
                            'PHE': 13, 
                            'PRO': 14,
                            'SER': 15,
                            'THR': 16,
                            'TRP': 17,
                            'TYR': 18, 
                            'VAL': 19,
                            'MSE': 12}

THREE_TO_ONE_TANKBIND = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

THREE_TO_ONE_DIFFDOCK = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # MSE this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}

ONE_TO_THREE_DIFFDOCK = {
                         'C': 'CYS',
                         'D': 'ASP', 
                         'S': 'SER',
                         'Q': 'GLN', 
                         'K': 'LYS', 
                         'I': 'ILE',
                         'P': 'PRO',
                         'T': 'THR',
                         'F': 'PHE',
                         'N': 'ASN',
                         'G': 'GLY',
                         'H': 'HIS',
                         'L': 'LEU',
                         'R': 'ARG',
                         'W': 'TRP',
                         'A': 'ALA',
                         'V': 'VAL',
                         'E': 'GLU',
                         'Y': 'TYR',
                         'M': 'MET'}

ONE_TO_NUMBER_DIFFDOCK = {THREE_TO_ONE_DIFFDOCK[three]: num for three, num in THREE_TO_NUMBER_DIFFDOCK.items()}

NUMBER_TO_ONE_DIFFDOCK = {num: one for one, num in ONE_TO_NUMBER_DIFFDOCK.items()}

ATOM_ORDER_DIFFDOCK = {'G': ['N', 'CA', 'C', 'O'],
                       'S': ['N', 'CA', 'C', 'O', 'CB', 'OG'],
                       'C': ['N', 'CA', 'C', 'O', 'CB', 'SG'],
                       'A': ['N', 'CA', 'C', 'O', 'CB'],
                       'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'],
                       'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
                       'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
                       'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
                       'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
                       'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
                       'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
                       'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
                       'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
                       'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
                       'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
                       'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],
                       'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
                       'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
                       'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],
                       'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2'],
                       'X': ['N', 'CA', 'C', 'O']}

CHI = { 'C' :
        { 1: ('N'  , 'CA' , 'CB' , 'SG' )   },
        'D' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'OD1'), },
        'E' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD' ),
          3: ('CB' , 'CG' , 'CD' , 'OE1'), },
        'F' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD1'), },
        'H' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'ND1'), },
        'I' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG1'),
          2: ('CA' , 'CB' , 'CG1', 'CD1'), },
        'K' :
        { 1: ('N'  , 'CA' , 'CB'  ,'CG' ),
          2: ('CA' , 'CB' , 'CG'  ,'CD' ),
          3: ('CB' , 'CG' , 'CD'  ,'CE' ),
          4: ('CG' , 'CD' , 'CE'  ,'NZ' ), },
        'L' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD1'), },
        'M' :
        { 1: ('N'  , 'CA' , 'CB'  ,'CG' ),
          2: ('CA' , 'CB' , 'CG'  ,'SD' ),
          3: ('CB' , 'CG' , 'SD'  ,'CE' ), },
        'N' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'OD1'), },
        'P' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD' ), },
        'Q' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD' ),
          3: ('CB' , 'CG' , 'CD' , 'OE1'), },
        'R' :
        { 1: ('N'  , 'CA' , 'CB'  ,'CG' ),
          2: ('CA' , 'CB' , 'CG'  ,'CD' ),
          3: ('CB' , 'CG' , 'CD'  ,'NE' ),
          4: ('CG' , 'CD' , 'NE'  ,'CZ' ), },
        'S' :
        { 1: ('N'  , 'CA' , 'CB' , 'OG' ), },
        'T' :
        { 1: ('N'  , 'CA' , 'CB' , 'OG1'), },
        'V' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG1'), },
        'W' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD1'), },
        'Y' :
        { 1: ('N'  , 'CA' , 'CB' , 'CG' ),
          2: ('CA' , 'CB' , 'CG' , 'CD1'), },
        }


ATOM_VOCAB_TANKBIND = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Cu",
    "Zn",
    "Se",
    "Br",
    "Sn",
    "I",
]
ATOM_VOCAB_TANKBIND = {a: i for i, a in enumerate(ATOM_VOCAB_TANKBIND)}
DEGREE_VOCAB = range(7)
NUM_HS_VOCAB = range(7)
FORMAL_CHARGE_VOCAB = range(-5, 6)
TOTAL_VALENCE_VOCAB = range(8)
BOND_TYPE_VOCAB = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_TYPE_VOCAB = {b: i for i, b in enumerate(BOND_TYPE_VOCAB)}
BOND_DIR_VOCAB = range(len(Chem.rdchem.BondDir.values))
BOND_STEREO_VOCAB = range(len(Chem.rdchem.BondStereo.values))
