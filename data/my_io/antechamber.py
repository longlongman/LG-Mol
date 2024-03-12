"""
reference code: https://github.com/pandegroup/vs-utils/blob/master/vs_utils/utils/amber_utils.py#L22
"""

import os
import shutil
import subprocess
import tempfile
from rdkit import Chem
import numpy as np
from collections import OrderedDict

class Antechamber(object):
    """
    Wrapper methods for Antechamber functionality.

    Calculations are carried out in a temporary directory because
    Antechamber writes out several files to disk.

    Parameters
    ----------
    charge_type : str, optional (default 'bcc')
        Antechamber charge type string. Defaults to AM1-BCC charges.
    """
    def __init__(self, charge_type='bcc'):
        self.charge_type = charge_type

        # temporary directory
        self.temp_dir = tempfile.mkdtemp()

    def __del__(self):
        """
        Cleanup.
        """
        shutil.rmtree(self.temp_dir)

    def get_charges_and_radii(self, mol):
        """
        Use Antechamber to calculate partial charges and atomic radii.

        Antechamber requires file inputs and output, so the molecule is
        written to SDF and Antechamber writes out a modified PDB (mpdb)
        containing charge and radius information.

        Note that Antechamber only processes the first molecule or
        conformer in the input file.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        net_charge = self.get_net_charge(mol)

        # write molecule to temporary file
        _, input_filename = tempfile.mkstemp(suffix='.sdf', dir=self.temp_dir)
        writer = Chem.SDWriter(input_filename)
        writer.write(mol)
        writer.close()

        # calculate charges and radii with Antechamber
        output_fd, output_filename = tempfile.mkstemp(suffix='.mpdb',
                                                      dir=self.temp_dir)
        os.close(output_fd)  # close temp file
        args = ['antechamber', '-i', input_filename, '-fi', 'sdf', '-o',
                output_filename, '-fo', 'mpdb', '-c', self.charge_type, '-nc',
                str(net_charge)]  # all arguments must be strings
        # args = [
        #     'antechamber',
        #     '-ek', "qm_theory='AM1'",
        #     '-i', input_filename,
        #     '-fi', 'sdf',
        #     '-o', output_filename,
        #     '-fo', 'mpdb',
        #     '-c', self.charge_type,
        #     '-nc', str(net_charge),
        #     '-j', '5',
        #     '-s', '2',
        #     '-dr', 'n'
        # ] # copy from UCFS ChimeraX

        try:
            subprocess.check_output(args, cwd=self.temp_dir)
        except subprocess.CalledProcessError as e:
            name = ''
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
            print("Antechamber: molecule '{}' failed.".format(name))
            with open(input_filename) as f:
                print(f.read())
            raise e

        # extract charges and radii
        reader = ModifiedPdbReader()
        with open(output_filename) as f:
            charges, radii = reader.get_charges_and_radii(f)

        return charges, radii

    @staticmethod
    def get_net_charge(mol):
        """
        Calculate the net charge on a molecule.

        Parameters
        ----------
        mol : RDMol
            Molecule.
        """
        net_charge = 0
        for atom in mol.GetAtoms():
            net_charge += atom.GetFormalCharge()
        return net_charge

class PdbReader(object):
    """
    Handle PDB files.

    Also supports conversion from PDB to Amber-style PQR files.
    """
    def parse_atom_record(self, line):
        """
        Extract fields from a PDB ATOM or HETATM record.

        See http://deposit.rcsb.org/adit/docs/pdb_atom_format.html.

        Parameters
        ----------
        line : str
            PDB ATOM or HETATM line.
        """
        assert line.startswith('ATOM') or line.startswith('HETATM')

        fields = OrderedDict()
        fields['record_name'] = line[:6]
        fields['serial_number'] = int(line[6:11])
        fields['atom_name'] = line[12:16]
        fields['alternate_location'] = line[16]
        fields['residue_name'] = line[17:20]
        fields['chain'] = line[21]
        fields['residue_number'] = int(line[22:26])
        fields['insertion_code'] = line[26]
        fields['x'] = float(line[30:38])
        fields['y'] = float(line[38:46])
        fields['z'] = float(line[46:54])

        # parse additional fields
        fields.update(self._parse_atom_record(line))

        # strip extra whitespace from fields
        for key in fields.keys():
            try:
                fields[key] = fields[key].strip()
            except AttributeError:
                pass

        return fields

    def _parse_atom_record(self, line):
        """
        Parse optional fields in ATOM and HETATM records.

        Parameters
        ----------
        line : str
            PDB ATOM or HETATM line.
        """
        fields = OrderedDict()
        try:
            fields['occupancy'] = float(line[54:60])
            fields['b_factor'] = float(line[60:66])
            fields['segment'] = line[72:76]
            fields['element'] = line[76:78]
            fields['charge'] = line[78:80]
        except IndexError:
            pass

        return fields

    def pdb_to_pqr(self, pdb, charges, radii):
        """
        Convert PDB to Amber-style PQR by adding charge and radius
        information. See p. 68 of the Amber 14 Reference Manual.

        Parameters
        ----------
        pdb : file_like
            PDB file.
        charges : array_like
            Atomic partial charges.
        radii : array_like
            Atomic radii.
        """

        # only certain PDB fields are used in the Amber PQR format
        pdb_fields = ['record_name', 'serial_number', 'atom_name',
                      'residue_name', 'chain', 'residue_number', 'x', 'y', 'z']

        i = 0
        pqr = ''
        for line in pdb:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                fields = self.parse_atom_record(line)

                # charge and radius are added after x, y, z coordinates
                pqr_fields = []
                for field in pdb_fields:
                    value = fields[field]
                    if value == '':
                        value = '?'
                    pqr_fields.append(str(value))
                pqr_fields.append(str(charges[i]))
                pqr_fields.append(str(radii[i]))
                line = ' '.join(pqr_fields) + '\n'
                i += 1  # update atom count
            pqr += line

        # check that we covered all the atoms
        assert i == len(charges) == len(radii)

        return pqr

class ModifiedPdbReader(PdbReader):
    """
    Handle Amber modified PDB files and generate Amber-style PQR files.
    """
    def _parse_atom_record(self, line):
        """
        Parse optional fields in ATOM and HETATM records.

        Amber modified PDB files contain charge, radius and atom type
        information in the fields following the x, y, z coordinates for
        atoms.

        Parameters
        ----------
        line : str
            Amber modified PDB ATOM or HETATM line.
        """
        fields = OrderedDict()
        charge, radius, amber_type = line[54:].strip().split()
        fields['charge'] = charge
        fields['radius'] = radius
        fields['amber_type'] = amber_type

        return fields

    def get_charges_and_radii(self, mpdb):
        """
        Extract atomic charges and radii from an Antechamber modified PDB
        file.

        Parameters
        ----------
        mpdb : file_like
            Antechamber modified PDB file.
        """
        charges = []
        radii = []
        for line in mpdb:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                fields = self.parse_atom_record(line)
                charges.append(fields['charge'])
                radii.append(fields['radius'])
        charges = np.asarray(charges, dtype=float)
        radii = np.asarray(radii, dtype=float)

        return charges, radii
