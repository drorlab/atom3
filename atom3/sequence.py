"""Code to do with pdb structure sequences."""
import cPickle as pickle
import os

import Bio
import Bio.SeqIO
import Bio.SeqRecord
import Bio.PDB.Polypeptide as poly
import pandas as pd

import atom3.structure as struct
import atom3.database as db


def write_fasta(seq_dict, fasta_out, id_out=None):
    """Given dictionary of name to sequence, write as FASTA."""
    records = []
    all_ids = {}
    for name, seq in seq_dict.items():
        residue_list, ids = residue_list_to_string(seq, with_ids=True)
        record = Bio.SeqRecord.SeqRecord(
            Bio.Seq.Seq(residue_list), id=name, description="")
        records.append(record)
        all_ids[name] = ids
    fasta_dir = os.path.dirname(os.path.abspath(fasta_out))
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)
    Bio.SeqIO.write(records, fasta_out, "fasta")
    if id_out is not None:
        id_dir = os.path.dirname(os.path.abspath(id_out))
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)
        with open(id_out, 'w') as f:
            pickle.dump(all_ids, f)


def pdb_to_fasta(pdb_filename, fasta_filename, id_filename, separate):
    """Write a pdb file as a fasta file."""
    flat_map = {}
    pdb_name = db.get_pdb_name(pdb_filename)
    structure = pd.read_pickle(pdb_filename)
    fasta_name_to_chain = {}
    for (chain, residues) in struct.get_chain_to_valid_residues(structure):
        fasta_name = pdb_name + '-' + chain[-2] + '-' + chain[-1]
        flat_map[fasta_name] = residues
        fasta_name_to_chain[fasta_name] = chain
    names = []
    filenames = []
    id_filenames = []
    if not separate:
        write_fasta(flat_map, fasta_filename, id_out=id_filename)
        filenames.append(fasta_filename)
        id_filenames.append(id_filename)
        names.append('all')
    else:
        for (name, seq) in flat_map.items():
            new_dict = {}
            new_dict[name] = seq
            filename = fasta_filename.format(name)
            filename2 = id_filename.format(name)
            write_fasta(new_dict, filename, id_out=filename2)
            names.append(fasta_name_to_chain[name])
            filenames.append(filename)
            id_filenames.append(filename2)
    return (names, filenames, id_filenames)


def residue_list_to_string(residues, with_ids=False):
    """Convert list of residues to string."""
    for residue in residues:
        if residue.resname == 'HID':
            residue.resname = 'HIS'
        elif residue.resname == 'CYX':
            residue.resname = 'CYS'
        elif residue.resname == 'ASX':
            residue.resname = 'ASP'
        elif residue.resname == 'GLX':
            residue.resname = 'GLY'

    seq = [poly.three_to_one(residue.resname) for residue in residues
           if residue.resname != 'SEC' and residue.resname != 'PYL']
    ids = [residue.residue for residue in residues
           if residue.resname != 'SEC' and residue.resname != 'PYL']
    if with_ids:
        return "".join(seq), ids
    else:
        return "".join(seq)
