"""
BioPython PDB Sructure wrapper class.

Allows for extraction of regions from the structure.
"""
import logging
import os
import re

import Bio.PDB
import Bio.PDB.Polypeptide as poly
import numpy as np
import pandas as pd

import atom3.database as db

# Max number of residues allowed.  Prevents processing of massive PDB
# structures.
max_residues = 100000


def extract_c_alpha_regions(structure, radius_ang, detailed=False):
    """
    Given a PDB filename, extracts biopython structure and then its regions.

    The regions are centered at the residue c-alphas.

    Args:
        pdb_filename (string):
            path to PDB file.
        radius_ang (int):
            radius of the surfacelets, in angstroms.

    Returns:
        Structured array of src.feat.region the regions extracted.
    """
    return structure.generate_regions(
        radius_ang,
        structure.generate_c_alpha_centroids(),
        detailed=detailed)


def parse_structure(structure_filename, concoord=False, one_model=False):
    """Parse a file into chain,model-to-residue mapping."""
    _, ext = os.path.splitext(structure_filename)
    detailed = ext == '.pkl'
    if detailed:
        # If detailed we are reading pandas pickle file outputted by
        # protprep.
        df = pd.read_pickle(structure_filename)
        # Set model to 0, because a multi-model file was either already split
        # into separate files (using the split command) or was pared down to a
        # single model by the autodock portion of the protprep pipeline.
        # This might need to be revisited if/when autodock is removed from
        # pipeline or we decide to actually keep track of correct model.
        df['model'] = get_model(structure_filename)
        # Remove hydrogens, for now, to maintain compatability.
        df = df[df['maestro_atom_name'].apply(lambda x: x.strip()[0]) != 'H']
    else:
        # BioPython.PDB Structure extracted from PDB file.
        biopy_structure = db.parse_biopython_structure(structure_filename)
        pdb_name = db.get_pdb_name(structure_filename)
        if concoord:
            # need to set model number to be correct (drawn from filename)
            # TODO: I (Raphael) moved this out of core Structure code, need to
            # make sure it is correct still for CONCOORD.
            biopy_structure = db.parse_biopython_structure(structure_filename)
            biopy_structure = \
                Bio.PDB.Structure.Structure(biopy_structure.id)

            chainmodel = pdb_name.split('_')[1]
            model_id = str(int(re.split('(\d+)', chainmodel)[1]) + 1)

            for model_obj in biopy_structure:
                new_model = Bio.PDB.Model.Model(model_id)
                for chain in model_obj:
                    new_model.add(chain)
                biopy_structure.add(new_model)

        if one_model:
            new_structure = Bio.PDB.Structure.Structure(biopy_structure.id)
            new_structure.add(biopy_structure[0])
            biopy_structure = new_structure
        atoms = []
        for residue in Bio.PDB.Selection.unfold_entities(biopy_structure, 'R'):
            # Prune out things that aren't actually residue atoms.
            if 'CA' in residue and residue.get_id()[0] == ' ':
                for atom in residue:
                    atoms.append(atom)

        df = pd.DataFrame([(
            pdb_name,
            str(atom.get_parent().get_parent().get_parent().serial_num),
            atom.get_parent().get_full_id()[2],
            str(atom.get_parent().get_id()[1]) +
            atom.get_parent().get_id()[2],
            atom.get_parent().get_resname(),
            atom.get_coord()[0],
            atom.get_coord()[1],
            atom.get_coord()[2],
            atom.get_id()[0],
            atom.get_name(),
            str(atom.serial_number)) for atom in atoms],
            columns=[
                'pdb_name',
                'model',
                'chain',
                'residue',
                'resname',
                'x',
                'y',
                'z',
                'element',
                'atom_name',
                'aid'])
    return df


def get_ca_pos_from_residues(df, res):
    """Look up alpha carbon positions of provided residues."""
    ca = df[df['atom_name'] == 'CA'].reset_index().set_index(
        ['pdb_name', 'model', 'chain', 'residue'])
    nb = ca.reindex(res)
    nb = nb.reset_index().set_index('index')
    return nb


def get_chain_to_valid_residues(structure, pdb_name=None):
    """Get tuples of chains and their valid residues."""
    if pdb_name is None:
        pdb_name = ()
    else:
        pdb_name = (pdb_name,)
    chain_res = []
    if type(structure) is Bio.PDB.Structure.Structure:
        for model in structure:
            for chain in model:
                residues = [res for res in chain
                            if poly.is_aa(res.get_resname(), standard=True) and
                            'CA' in res]
                if len(residues) != 0:
                    chain_res.append((
                        pdb_name + (str(model.serial_num), chain.get_id()),
                        residues))
    else:
        if 'atom_name' in structure.columns:
            calphas = structure[structure['atom_name'] == 'CA']
        else:
            calphas = structure[structure['maestro_atom_name'] == ' CA ']
        calphas = calphas[calphas['resname'] != 'UNK']

        for (chain, chain_ca) in calphas.groupby(['model', 'chain']):
            residues = [ca for idx, ca in chain_ca.iterrows()]
            if len(residues) != 0:
                chain_res.append((pdb_name + chain, residues))
    return chain_res


def df_to_ca(df):
    """Extract only alpha carbons from dataframe."""
    ca = df[df['atom_name'] == 'CA']
    ca.index.name = 'df_index'
    ca = ca.reset_index()
    return ca


def create_lookup(ca):
    lookup = {}

    def lookup_aa_id(pdb_name, model, chain, residue):
        """Lookup amino acid id based on pdb_name, model, chain, residue id."""
        if (pdb_name, model, chain, residue) not in lookup:
            res = ca.loc[
                (ca['residue'] == residue) &
                (ca['chain'] == chain) &
                (ca['model'] == model) &
                (ca['pdb_name'] == pdb_name)].index
            assert len(res) == 1
            lookup[(pdb_name, model, chain, residue)] = res[0]

        return lookup[(pdb_name, model, chain, residue)]
    return lookup_aa_id


expected = {
    "TRP": 14,
    "PHE": 11,
    "LYS": 9,
    "PRO": 7,
    "ASP": 8,
    "ALA": 5,
    "ARG": 11,
    "CYS": 6,
    "VAL": 7,
    "THR": 7,
    "GLY": 4,
    "SER": 6,
    "HIS": 10,
    "LEU": 8,
    "GLU": 9,
    "TYR": 12,
    "ILE": 8,
    "ASN": 8,
    "MET": 8,
    "GLN": 9,
}


def add_sidechains_parser(subparsers, pp):
    """Add parser."""

    def get_missing_sidechains_main(args):
        get_missing_sidechains(args.pdb_dataset, args.output_scwrl)

    ap = subparsers.add_parser(
        'sidechains', description="find missing side chains",
        help='generates a sequence for use with scwrl4 where missing side'
        ' residues are upper-cased',
        parents=[pp])
    ap.set_defaults(func=get_missing_sidechains_main)
    ap.add_argument('pdb_dataset', type=str,
                    help='pdb dataset to get missing side chains for.')
    ap.add_argument('output_scwrl', type=str,
                    help='where to write output sequences to.')


def get_missing_sidechains(pdb_dataset, output_scwrl):
    """Get residues that are missing atoms."""
    for pdb_filename in db.get_structures_filenames(pdb_dataset):
        biopy_structure = db.parse_biopython_structure(pdb_filename)
        pdb_name = db.get_pdb_name(pdb_filename)
        missing = 0
        scwrl_list = []
        logging.info("Processing {:}".format(pdb_name))
        for model in biopy_structure:
            for chain in model:
                for i, residue in enumerate(chain):
                    res_name = residue.resname
                    if res_name not in expected:
                        logging.warning("Non-standard residue found: {:}. "
                                        "Skipping.".format(res_name))
                        continue
                    res_code = poly.three_to_one(res_name)
                    res_id = residue.id[1]
                    curr_count = len(
                        Bio.PDB.Selection.unfold_entities(residue, 'A'))
                    if curr_count != expected[res_name]:
                        logging.debug(
                            "Missing residue {:} at position {:} (with id {:})"
                            " which has {:} instead of the expected {:} atoms."
                            .format(res_name, i, res_id, curr_count,
                                    expected[res_name]))
                        missing += 1
                        scwrl_list.append(res_code.upper())
                    else:
                        scwrl_list.append(res_code.lower())

        logging.debug("Missing {:} residue total".format(missing))
        with open(output_scwrl, 'w') as f:
            f.write("".join(scwrl_list))


def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    v = df.values
    cols = df.columns

    types = [(cols[i].encode(), df[k].dtype.type)
             for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z


def get_model(pdb_filename):
    """Get model from split pdb filename."""
    pdb_name = db.get_pdb_name(pdb_filename, with_type=False)
    tokens = pdb_name.split("_")
    if len(tokens) < 3:
        return 0
    elif not tokens[2].isdigit():
        return 0
    else:
        return int(tokens[2])


def get_chain(pdb_filename):
    """Get chain from split pdb filename."""
    pdb_name = db.get_pdb_name(pdb_filename, with_type=False)
    tokens = pdb_name.split("_")
    if len(tokens) < 3:
        return 0
    else:
        return tokens[1]


def df_to_pdb(df_in):
    """Convert df to pdb."""
    df = df_in.copy()
    new_structure = Bio.PDB.Structure.Structure('')
    for (model, m_atoms) in df.groupby(['model']):
        new_model = Bio.PDB.Model.Model(model)
        for (chain, c_atoms) in m_atoms.groupby(['chain']):
            new_chain = Bio.PDB.Chain.Chain(chain)
            for (residue, r_atoms) in c_atoms.groupby(['residue']):
                new_residue = residue_to_pdbresidue(residue, r_atoms)
                new_chain.add(new_residue)
            new_model.add(new_chain)
        new_structure.add(new_model)
    return new_structure


def residue_to_pdbresidue(residue, r_atoms):
    resname = r_atoms['resname'].unique()
    assert len(resname) == 1
    resname = resname[0]
    new_residue = Bio.PDB.Residue.Residue(
        (' ', int(residue[:-1]), residue[-1]), resname, '')
    for row, atom in r_atoms.iterrows():
        new_atom = Bio.PDB.Atom.Atom(
            atom['atom_name'],
            [atom['x'], atom['y'], atom['z']],
            1,
            1,
            ' ',
            atom['atom_name'],
            atom['aid'],
            atom['element'])
        new_residue.add(new_atom)
    return new_residue
