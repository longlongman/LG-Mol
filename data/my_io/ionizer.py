"""
reference code: https://github.com/pandegroup/vs-utils/blob/master/vs_utils/utils/ob_utils.py#L17
"""

from rdkit import Chem
import subprocess
import serial
from io import StringIO, BytesIO

class IonizerError(Exception):
    """
    Generic Ionizer exception.
    """

class Ionizer(object):
    """
    Calculate atomic formal charges at the given pH.

    Parameters
    ----------
    pH : float, optional (default 7.4)
        pH at which to calculate formal charges.
    """
    def __init__(self, pH=7.4):
        self.pH = pH

    def __call__(self, mol):
        """
        Ionize a molecule.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        return self.ionize(mol)

    def ionize(self, mol):
        """
        Ionize a molecule while preserving 3D coordinates.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        if mol.GetNumConformers() > 0:
            return self._ionize_3d(mol)
        else:
            return self._ionize_2d(mol)

    def _ionize_2d(self, mol):
        """
        Ionize a molecule without preserving conformers.

        Note: this method removes explicit hydrogens from the molecule.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        args = ['obabel', '-i', 'can', '-o', 'can', '-p', str(self.pH)]
        p = subprocess.Popen(args, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        ionized_smiles, _ = p.communicate(smiles)
        ionized_mol = Chem.MolFromSmiles(ionized_smiles)

        # catch ionizer error
        if ionized_mol is None:
            raise IonizerError(mol)

        return ionized_mol

    def _ionize_3d(self, mol):
        """
        Ionize a molecule while preserving conformers.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        assert mol.GetNumConformers() > 0
        sdf = ''
        for conf in mol.GetConformers():
            sdf += Chem.MolToMolBlock(mol, confId=conf.GetId(),
                                      includeStereo=True)
            sdf += '$$$$\n'
        args = ['obabel', '-i', 'sdf', '-o', 'sdf', '-p', str(self.pH)]
        p = subprocess.Popen(args, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        
        # ionized_sdf, _ = p.communicate(sdf)
        ionized_sdf, _ = p.communicate(sdf.encode('utf-8'))
        
        # reader = serial.MolReader(StringIO(ionized_sdf), mol_format='sdf',
        #                           remove_salts=False)  # no changes
        reader = serial.MolReader(BytesIO(ionized_sdf), mol_format='sdf',
                                  remove_salts=False)  # no changes
        try:
            mols = list(reader.get_mols())
        except RuntimeError as e:  # catch pre-condition violations
            raise IonizerError(e.message)

        # catch ionizer failure
        if len(mols) == 0:
            raise IonizerError(mol)

        # detection of stereochemistry based on 3D coordinates might result
        # in issues when attempting to recombine ionized conformers, but we
        # merge them anyway
        if len(mols) == 1:
            ionized_mol, = mols
        else:
            ionized_mol = mols[0]
            for other in mols[1:]:
                for conf in other.GetConformers():
                    ionized_mol.AddConformer(conf, assignId=True)
        return ionized_mol

if __name__ == '__main__':
    ionizer = Ionizer(pH=7.4)
    mol = Chem.SDMolSupplier('/sharefs/longsiyu/projects/shape2mol/data/playground/2rd6_B_78P.sdf')[0]
    ionizer(mol)