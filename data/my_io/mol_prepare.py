"""
reference code: https://github.com/pandegroup/vs-utils/blob/master/vs_utils/features/__init__.py#L256
"""

from ionizer import Ionizer
from rdkit import Chem

class MolPreparator(object):
    """
    Molecule preparation for calculating features.

    Parameters
    ----------
    ionize : bool, optional (default False)
        Whether to ionize molecules.
    pH : float, optional (default 7.4)
        Ionization pH.
    align : bool, optional (default False)
        Whether to canonicalize the orientation of molecules. This requires
        removal and readdition of hydrogens. This is usually not required
        when working with conformers retrieved from PubChem.
    add_hydrogens : bool, optional (default False)
        Whether to add hydrogens (with coordinates) to molecules.
    """
    def __init__(self, ionize=False, pH=7.4, align=False, add_hydrogens=False):
        self.ionize = ionize
        self.ionizer = Ionizer(pH)
        self.align = align
        self.add_hydrogens = add_hydrogens
    
    def __call__(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)
    
    def set_ionize(self, ionize):
        """
        Set ionization flag.

        Parameters
        ----------
        ionize : bool
            Whether to ionize molecules.
        """
        self.ionize = ionize
    
    def set_pH(self, pH):
        """
        Set ionization pH.

        Parameters
        ----------
        value : float
            Ionization pH.
        """
        self.ionizer = Ionizer(pH)

    def set_align(self, align):
        """
        Set align flag.

        Parameters
        ----------
        align : bool
            Whether to align molecules.
        """
        self.align = align

    def set_add_hydrogens(self, add_hydrogens):
        """
        Set add_hydrogens flag.

        Parameters
        ----------
        add_hydrogens : bool
            Whether to add hydrogens.
        """
        self.add_hydrogens = add_hydrogens
    
    def prepare(self, mol, ionize=None, align=None, add_hydrogens=None):
        """
        Prepare a molecule for featurization.

        Default values for individual steps can be overriden with keyword
        arguments. For example, to disable ionization for a specific molecule,
        include ionize=False.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        ionize : bool, optional (default None)
            Override for self.ionize.
        align : bool, optional (default None)
            Override for self.align.
        add_hydrogens : bool, optional (default None)
            Override for self.add_hydrogens.
        """
        if ionize is None:
            ionize = self.ionize
        if align is None:
            align = self.align
        if add_hydrogens is None:
            add_hydrogens = self.add_hydrogens

        mol = Chem.Mol(mol)  # create a copy

        # ionization
        if ionize:
            mol = self.ionizer(mol)

        # orientation
        if align:

            # canonicalization can fail when hydrogens are present
            mol = Chem.RemoveHs(mol)
            center = rdGeometry.Point3D(0, 0, 0)
            for conf in mol.GetConformers():
                rdMolTransforms.CanonicalizeConformer(conf, center=center)

        # hydrogens
        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=True)
        return mol