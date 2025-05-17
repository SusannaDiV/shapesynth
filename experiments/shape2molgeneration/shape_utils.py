import numpy as np
from rdkit.Chem import rdMolTransforms
from math import ceil, pi
import random
import copy
from skimage.util import view_as_blocks
# van der Waals radius
ATOM_RADIUS = {
    'C': 1.908,
    'F': 1.75,
    'Cl': 1.948,
    'Br': 2.22,
    'I': 2.35,
    'N': 1.824,
    'O': 1.6612,
    'P': 2.1,
    'S': 2.0,
    'Si': 2.2, # not accurate
    'H': 1.0
}

# atomic number
ATOMIC_NUMBER = {
    'C': 6,
    'F': 9,
    'Cl': 17,
    'Br': 35,
    'I': 53,
    'N': 7,
    'O': 8,
    'P': 15,
    'S': 16,
    'Si': 14,
    'H': 1
}

ATOMIC_NUMBER_REVERSE = {v: k for k, v in ATOMIC_NUMBER.items()}
def get_mol_centroid(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = np.mean(conformer.GetPositions(), axis=0)
    return centroid

def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation

def centralize(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    centroid = get_mol_centroid(mol, confId)
    translation = trans(-centroid[0], -centroid[1], -centroid[2])  
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol


import pickle

import numpy as np

import pybel
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
from collections import defaultdict


class Featurizer():
    """Calcaulates atomic features for molecules. Features can encode atom type,
    native pybel properties or any property defined with SMARTS patterns

    Attributes
    ----------
    FEATURE_NAMES: list of strings
        Labels for features (in the same order as features)
    NUM_ATOM_CLASSES: int
        Number of atom codes
    ATOM_CODES: dict
        Dictionary mapping atomic numbers to codes
    NAMED_PROPS: list of string
        Names of atomic properties to retrieve from pybel.Atom object
    CALLABLES: list of callables
        Callables used to calculcate custom atomic properties
    SMARTS: list of SMARTS strings
        SMARTS patterns defining additional atomic properties
    """

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):

        """Creates Featurizer with specified types of features. Elements of a
        feature vector will be in a following order: atom type encoding
        (defined by atom_codes), Pybel atomic properties (defined by
        named_properties), molecule code (if present), custom atomic properties
        (defined `custom_properties`), and additional properties defined with
        SMARTS (defined with `smarts_properties`).

        Parameters
        ----------
        atom_codes: dict, optional
            Dictionary mapping atomic numbers to codes. It will be used for
            one-hot encoging therefore if n different types are used, codes
            shpuld be from 0 to n-1. Multiple atoms can have the same code,
            e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
            and nitrogens and oxygens with [0, 1] vectors. If not provided,
            default encoding is used.
        atom_labels: list of strings, optional
            Labels for atoms codes. It should have the same length as the
            number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
            should provide something like ['C', 'O or N']. If not specified
            labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
            specified this argument is ignored.
        named_properties: list of strings, optional
            Names of atomic properties to retrieve from pybel.Atom object. If
            not specified ['hyb', 'heavyvalence', 'heterovalence',
            'partialcharge'] is used.
        save_molecule_codes: bool, optional (default True)
            If set to True, there will be an additional feature to save
            molecule code. It is usefeul when saving molecular complex in a
            single array.
        custom_properties: list of callables, optional
            Custom functions to calculate atomic properties. Each element of
            this list should be a callable that takes pybel.Atom object and
            returns a float. If callable has `__name__` property it is used as
            feature label. Otherwise labels 'func<i>' etc are used, where i is
            the index in `custom_properties` list.
        smarts_properties: list of strings, optional
            Additional atomic properties defined with SMARTS patterns. These
            patterns should match a single atom. If not specified, deafult
            patterns are used.
        smarts_labels: list of strings, optional
            Labels for properties defined with SMARTS. Should have the same
            length as `smarts_properties`. If not specified labels 'smarts0',
            'smarts1' etc are used. If `smarts_properties` is not specified
            this argument is ignored.
        """

        # Remember namse of all features in the correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavyvalence', 'heterovalence',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    # 9 classes
                    #   B/C/N/卤素/金属...
                    self.encode_num(atom.atomicnum),
                    # 4 classes
                    #   hyb=杂化方式 heavyvalence=连接的非H原子数量
                    #   heterovalence=连接的非CH原子数量 partialcharge=部分电荷
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))

        coords = np.array(coords, dtype=np.float64)
        features = np.array(features, dtype=np.float64)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              # 5 classes
                              #   基于SMARTS表达式提取出的一些信息，例如环、芳香环等
                              self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer


def rotation_matrix(axis, theta):
    """Counterclockwise rotation about a given axis by theta radians"""

    if not isinstance(axis, (np.ndarray, list, tuple)):
        raise TypeError('axis must be an array of floats of shape (3,)')
    try:
        axis = np.asarray(axis, dtype=np.float64)
    except ValueError:
        raise ValueError('axis must be an array of floats of shape (3,)')

    if axis.shape != (3,):
        raise ValueError('axis must be an array of floats of shape (3,)')

    if not isinstance(theta, (float, int)):
        raise TypeError('theta must be a float')

    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# Create matrices for all possible 90* rotations of a box
ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

# about X, Y and Z - 9 rotations
for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

# about each face diagonal - 6 rotations
for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

# about each space diagonal - 8 rotations
for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))


def rotate(coords, rotation):
    """Rotate coordinates by a given rotation

    Parameters
    ----------
    coords: array-like, shape (N, 3)
        Arrays with coordinates and features for each atoms.
    rotation: int or array-like, shape (3, 3)
        Rotation to perform. You can either select predefined rotation by
        giving its index or specify rotation matrix.

    Returns
    -------
    coords: np.ndarray, shape = (N, 3)
        Rotated coordinates.
    """

    global ROTATIONS

    if not isinstance(coords, (np.ndarray, list, tuple)):
        raise TypeError('coords must be an array of floats of shape (N, 3)')
    try:
        coords = np.asarray(coords, dtype=np.float64)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    shape = coords.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    if isinstance(rotation, int):
        if rotation >= 0 and rotation < len(ROTATIONS):
            return np.dot(coords, ROTATIONS[rotation])
        else:
            raise ValueError('Invalid rotation number %s!' % rotation)
    elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
        return np.dot(coords, rotation)

    else:
        raise ValueError('Invalid rotation %s!' % rotation)


def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
    """Convert atom coordinates and features represented as 2D arrays into a
    fixed-sized 3D box.

    所有涉及距离的单位都是angstrom(0.1nm)，则：
    - 假设grid_resoltution=1，max_dist=10，则构建一个20x20x20的立方体，其中每个立方体的边长为1。
    - coords是Nx3的原子位置矩阵（已减均值），则将其排布在立方体中，每个原子所处的格子设置为1。
        int((coords+max_dist)/grid_resolution)
    - features是Nx18的原子feature矩阵，将每个原子对应的feature填在格子里。
    返回20x20x20x18的tensor。

    Parameters
    ----------
    coords, features: array-likes, shape (N, 3) and (N, F)
        Arrays with coordinates and features for each atoms.
    grid_resolution: float, optional
        Resolution of a grid (in Angstroms).
    max_dist: float, optional
        Maximum distance between atom and box center. Resulting box has size of
        2*`max_dist`+1 Angstroms and atoms that are too far away are not
        included.

    Returns
    -------
    coords: np.ndarray, shape = (M, M, M, F)
        4D array with atom properties distributed in 3D space. M is equal to
        2 * `max_dist` / `grid_resolution` + 1
    """

    try:
        coords = np.asarray(coords, dtype=np.float64)
    except ValueError:
        raise ValueError('coords must be an array of floats of shape (N, 3)')
    c_shape = coords.shape
    if len(c_shape) != 2 or c_shape[1] != 3:
        raise ValueError('coords must be an array of floats of shape (N, 3)')

    N = len(coords)
    try:
        features = np.asarray(features, dtype=np.float64)
    except ValueError:
        raise ValueError('features must be an array of floats of shape (N, F)')
    f_shape = features.shape
    if len(f_shape) != 2 or f_shape[0] != N:
        raise ValueError('features must be an array of floats of shape (N, F)')

    if not isinstance(grid_resolution, (float, int)):
        raise TypeError('grid_resolution must be float')
    if grid_resolution <= 0:
        raise ValueError('grid_resolution must be positive')

    if not isinstance(max_dist, (float, int)):
        raise TypeError('max_dist must be float')
    if max_dist <= 0:
        raise ValueError('max_dist must be positive')

    num_features = f_shape[1]
    max_dist = float(max_dist)
    grid_resolution = float(grid_resolution)

    box_size = ceil(2 * max_dist / grid_resolution + 1)

    # move all atoms to the neares grid point
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)

    # remove atoms outside the box
    in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
    grid = np.zeros((1, box_size, box_size, box_size, num_features),
                    dtype=np.float64)
    
    f_grid = defaultdict(list)
    for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
        grid[0, x, y, z] += f
        f_grid[tuple(f)].append((x, y, z))

    return grid, f_grid




def get_atom_stamp(grid_resolution, max_dist):
    # atom stamp is a sphere which radius equal to atom van der Waals radius
    def _get_atom_stamp(symbol):
        box_size = ceil(2 * max_dist // grid_resolution + 1)

        x, y, z = np.indices((box_size, box_size, box_size))
        x = x * grid_resolution + grid_resolution / 2
        y = y * grid_resolution + grid_resolution / 2
        z = z * grid_resolution + grid_resolution / 2

        mid = (box_size // 2, box_size // 2, box_size // 2)
        mid_x = x[mid]
        mid_y = y[mid]
        mid_z = z[mid]

        sphere = (x - mid_x)**2 + (y - mid_y)**2 + (z - mid_z)**2 \
            <= ATOM_RADIUS[symbol]**2
        sphere = sphere.astype(int)
        sphere[sphere > 0] = ATOMIC_NUMBER[symbol]
        return sphere

    atom_stamp = {}
    for symbol in ATOM_RADIUS:
        atom_stamp[symbol] = _get_atom_stamp(symbol)
    return atom_stamp

# def get_binary_features(mol, confId):
#     coords = []
#     features = []
#     confermer = mol.GetConformer(confId)
#     for atom in mol.GetAtoms():
#         idx = atom.GetIdx()
#         coord = list(confermer.GetAtomPosition(idx))
#         coords.append(coord)
#         features.append(atom.GetAtomicNum())
#     coords = np.array(coords)
#     features = np.array(features)
#     features = np.expand_dims(features, axis=1)
#     return coords, features

def get_atom_prop(atom, prop_name):
    if atom.HasProp(prop_name):
        return atom.GetProp(prop_name)
    else:
        return None

def get_binary_features(mol, confId, without_H):
    coords = []
    features = []
    confermer = mol.GetConformer(confId)
    for atom in mol.GetAtoms():
        if atom.HasProp('mask') and get_atom_prop(atom, 'mask') == 'true':
            continue
        idx = atom.GetIdx()
        syb = atom.GetSymbol()
        if without_H and syb == 'H':
            continue
        coord = list(confermer.GetAtomPosition(idx))
        coords.append(coord)
        features.append(atom.GetAtomicNum())
    coords = np.array(coords, dtype=np.float64)
    features = np.array(features)
    features = np.expand_dims(features, axis=1)
    return coords, features

# def get_shape(mol, atom_stamp, grid_resolution, max_dist, confId=-1):
#     # expand each atom point to a sphere
#     coords, features = get_binary_features(mol, confId)
#     grid, atomic2grid = make_grid(coords, features, grid_resolution, max_dist)
#     shape = np.zeros(grid[0, :, :, :, 0].shape)
#     for tup in atomic2grid:
#         atomic_number = int(tup[0])
#         stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
#         for grid_ijk in atomic2grid[tup]:
#             i = grid_ijk[0]
#             j = grid_ijk[1]
#             k = grid_ijk[2]

#             x_left = i - stamp.shape[0] // 2 if i - stamp.shape[0] // 2 > 0 else 0
#             x_right = i + stamp.shape[0] // 2 if i + stamp.shape[0] // 2 < shape.shape[0] else shape.shape[0] - 1
#             x_l = i - x_left
#             x_r = x_right - i

#             y_left = j - stamp.shape[1] // 2 if j - stamp.shape[1] // 2 > 0 else 0
#             y_right = j + stamp.shape[1] // 2 if j + stamp.shape[1] // 2 < shape.shape[1] else shape.shape[1] - 1
#             y_l = j - y_left
#             y_r = y_right - j

#             z_left = k - stamp.shape[2] // 2 if k - stamp.shape[2] // 2 >0 else 0
#             z_right = k + stamp.shape[2] // 2 if k + stamp.shape[2] // 2 < shape.shape[2] else shape.shape[2] - 1
#             z_l = k - z_left
#             z_r = z_right - k

#             mid = stamp.shape[0] // 2
#             shape_part =  shape[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
#             stamp_part = stamp[mid - x_l: mid + x_r + 1, mid - y_l: mid + y_r + 1, mid - z_l: mid + z_r + 1]

#             shape_part += stamp_part
#     shape[shape > 0] = 1
#     return shape

def get_shape(mol, atom_stamp, grid_resolution, max_dist, confId=-1, without_H=True, by_coords=False, coords=None, features=None):
    # expand each atom point to a sphere
    if not by_coords:
        coords, features = get_binary_features(mol, confId, without_H)
    grid, atomic2grid = make_grid(coords, features, grid_resolution, max_dist)
    shape = np.zeros(grid[0, :, :, :, 0].shape)
    for tup in atomic2grid:
        atomic_number = int(tup[0])
        stamp = atom_stamp[ATOMIC_NUMBER_REVERSE[atomic_number]]
        for grid_ijk in atomic2grid[tup]:
            i = grid_ijk[0]
            j = grid_ijk[1]
            k = grid_ijk[2]

            x_left = i - stamp.shape[0] // 2 if i - stamp.shape[0] // 2 > 0 else 0
            x_right = i + stamp.shape[0] // 2 if i + stamp.shape[0] // 2 < shape.shape[0] else shape.shape[0] - 1
            x_l = i - x_left
            x_r = x_right - i

            y_left = j - stamp.shape[1] // 2 if j - stamp.shape[1] // 2 > 0 else 0
            y_right = j + stamp.shape[1] // 2 if j + stamp.shape[1] // 2 < shape.shape[1] else shape.shape[1] - 1
            y_l = j - y_left
            y_r = y_right - j

            z_left = k - stamp.shape[2] // 2 if k - stamp.shape[2] // 2 >0 else 0
            z_right = k + stamp.shape[2] // 2 if k + stamp.shape[2] // 2 < shape.shape[2] else shape.shape[2] - 1
            z_l = k - z_left
            z_r = z_right - k

            mid = stamp.shape[0] // 2
            shape_part =  shape[x_left: x_right + 1, y_left: y_right + 1, z_left: z_right + 1]
            stamp_part = stamp[mid - x_l: mid + x_r + 1, mid - y_l: mid + y_r + 1, mid - z_l: mid + z_r + 1]

            shape_part += stamp_part
    shape[shape > 0] = 1
    return shape

def sample_augment(sample, rotation_bin, max_translation, confId=-1):
    sample = copy.deepcopy(sample)
    confermer = sample['mol'].GetConformer(confId)

    rot = random.choice(range(rotation_bin))
    rotation_mat = ROTATIONS[rot]

    # rotation the molecule
    rotation = np.zeros((4, 4))
    rotation[:3, :3] = rotation_mat
    rdMolTransforms.TransformConformer(confermer, rotation)

    # rotation fragments
    for fragment in sample['frag_list']:
        frag_rotation_mat = fragment['rotate_mat']
        frag_trans_vec = fragment['trans_vec']
        
        frag_rotation_translation = np.zeros((4, 4))
        frag_rotation_translation[:3, :3] = frag_rotation_mat
        frag_rotation_translation[:3, 3] = frag_trans_vec

        frag_rotation_translation_rotation = np.dot(rotation, frag_rotation_translation)

        fragment['rotate_mat'] = frag_rotation_translation_rotation[:3, :3]
        fragment['trans_vec'] = frag_rotation_translation_rotation[:3, 3]

    tr = max_translation * np.random.rand(3)

    # translate the molecule
    translate = trans(tr[0], tr[1], tr[2])
    rdMolTransforms.TransformConformer(confermer, translate)

    # translate fragments
    for fragment in sample['frag_list']:
        frag_trans_vec = fragment['trans_vec']
        frag_trans_vec = frag_trans_vec + tr
        fragment['trans_vec'] = frag_trans_vec

    return sample

def get_shape_patches(shape, patch_size):
    assert shape.shape[0] % patch_size == 0
    shape_patches = view_as_blocks(shape, (patch_size, patch_size, patch_size))
    return shape_patches

def time_shift(s):
    return s[:-1], s[1:]

def get_grid_coords(coords, max_dist, grid_resolution):
    grid_coords = (coords + max_dist) / grid_resolution
    grid_coords = grid_coords.round().astype(int)
    return grid_coords

def get_rotation_bins(sp, rp):
    mid = sp // 2
    sr = 1.0 / sp

    face1 = []
    for y in range(sp):
        for z in range(sp):
            face1.append(np.array([0.5, (y - mid) * sr, (z - mid) * sr]))
    face2 = []
    for x in range(sp):
        for y in range(sp):
            face2.append(np.array([(x - mid) * sr, (y - mid) * sr, 0.5]))
    face3 = []
    for x in range(sp):
        for z in range(sp):
            face3.append(np.array([(x - mid) * sr, 0.5, (z - mid) * sr]))
    
    face_point = face1 + face2 + face3
    
    rotation_mat_bin = [rotation_matrix(np.array((1, 1, 1)), 0)]
    for p in face_point:
        for t in range(1, rp):
            axis = p
            theta = t * pi / (rp / 2)
            rotation_mat_bin.append(rotation_matrix(axis, theta))
    rotation_mat_bin = np.stack(rotation_mat_bin, axis=0)

    return rotation_mat_bin
