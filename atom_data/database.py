"""All functions related to processing PDB datasets."""
import collections as col
import logging

import glob
import gzip
import os.path

import Bio.PDB
import numpy as np


def identity(x):
    """Return input."""
    return x


def always(x):
    """Always return true."""
    return True


def get_unique_pdbs(pdb_dataset, uniquer=identity, extension=None,
                    keeper=always):
    """
    Get pdbs in dataset that are unique according to uniquer.

    Store them in dict mapping from unique key to list of pdbs.  Keeper is
    function which determines whether we want to use the provided file.
    """
    unique_pdbs = col.defaultdict(list)
    for pdb_filename in get_structures_filenames(pdb_dataset,
                                                 extension=extension):
        if keeper(pdb_filename):
            key = uniquer(pdb_filename)
            unique_pdbs[key].append(pdb_filename)
    return unique_pdbs


def parse_biopython_structure(pdb_filename):
    """
    Extract a Bio.PDB structure from the provided PDB file.

    Args:
        pdb_filename (string):
            file path to PDB file we wish to extract Biopython structure from.

    Returns:
        Bio.PDB.Structure containing the file's data.

    Raises:
        ValueError:
            if filetype of pdb_filename not recognized.
    """
    parser = Bio.PDB.PDBParser(QUIET=True)
    _, ext = os.path.splitext(pdb_filename)

    if ext == ".gz":
        biopy_structure = parser.get_structure('pdb', gzip.open(pdb_filename))
    elif ".pdb" in ext:
        biopy_structure = parser.get_structure('pdb', pdb_filename)
    else:
        raise ValueError("Unrecognized filetype " + pdb_filename)
    return biopy_structure


def get_pdb_group(pdb_filename):
    """
    Get 2 letter grouping from pdb filename.

    Grab the 2 middle letters of the pdb code. E.g. 2olx pdb name gives ol as
    the group.
    """
    return get_pdb_code(pdb_filename)[1:3]


def get_pdb_basename(pdb_filename):
    """
    Get 4 letter code from pdb filename suffixed by pdb type.

    E.g. test_dir/11as_l_u_unbound.pdb1.gz has a basename of 11as.pdb1 .
    """
    return get_pdb_code(pdb_filename) + '.' + get_pdb_type(pdb_filename)


def get_pdb_type(pdb_filename):
    """
    Get type of pdb, as in extension.

    E.g. test_dir/11as.pdb1 has a type of pdb1 .
    """
    return get_pdb_name(pdb_filename).split('.')[1]


def get_pdb_code(pdb_filename):
    """
    Get 4 letter code from pdb filename.

    E.g. test_dir/11as.pdb1 has a code of 11as .
    """
    pdb_name = get_pdb_name(pdb_filename)
    if pdb_name.startswith("complex_"):
        # Haddock style name.
        return pdb_name.split('.')[0].split('_')[-1]
    else:
        return get_pdb_name(pdb_filename)[0:4]


def get_pdb_name(pdb_filename, with_type=True):
    """
    Get pdb name from full file path.

    Args:
        filename (string):
            path to pdb file.

    Returns:
        pdb name, e.g. 1aa1.pdb1
    """
    if with_type:
        return '.'.join(os.path.basename(pdb_filename).split('.')[0:2])
    else:
        return os.path.basename(pdb_filename).split('.')[0]


def get_filenames(pdb_name, pdb_dataset, extension=None, enforcement='one'):
    """
    Get path to pdb file, given pdb name and path to pdb dataset.

    Args:
        pdb_name (string):
            name of pdb file, e.g. 1aa1.pdb1
        pdb_dataset (string):
            path to pdb dataset.

    Returns:
        path to file.  If multiple matches, return first one.
    """
    filenames = get_structures_filenames(pdb_dataset, extension=extension)
    good = [x for x in filenames if pdb_name in x]
    _check_filenames(pdb_name, good, enforcement=enforcement)
    return good


def get_all_filenames(pdb_names, pdb_dataset, extension=None,
                      keyer=get_pdb_name, enforcement='one'):
    """Return filenames for provided pdb names, in same order."""
    filenames = get_structures_filenames(pdb_dataset, extension=extension)
    dict = col.defaultdict(list)
    for filename in filenames:
        dict[keyer(filename)].append(filename)
    res = []
    for pdb_name in pdb_names:
        good = dict[pdb_name]
        _check_filenames(pdb_name, good, enforcement=enforcement)
        res.append(good)
    return res


def _check_filenames(pdb_name, pdb_filenames, enforcement='one'):
    """Make sure we have the right number of filenames."""
    if enforcement == 'one' and len(pdb_filenames) != 1:
        logging.error(
            "Need exactly one file per key, instead for {:} got {:} ({:})"
            .format(pdb_name, len(pdb_filenames), pdb_filenames))
        raise RuntimeError
    elif enforcement == 'at_least_one' and len(pdb_filenames) < 1:
        logging.error(
            "Need at least one file per key, instead for {:} got {:} "
            "({:})"
            .format(pdb_name, len(pdb_filenames), pdb_filenames))
        raise RuntimeError
    elif enforcement == 'none':
        pass


def get_structures_filenames(pdb_dataset, extension=None):
    """
    Return structure filenames from PDB database specified in bio_dir file.

    Args:
        bio_dir (string):
            location of PDB snapshot's biounit divided directory (i.e.
            snapshot/data/biounit/PDB/divided)
    """
    # Cache snapshot filenames here since a bit expensive.

    if _is_pdb_list(pdb_dataset):
        return pdb_dataset

    if _is_pdb_regex(pdb_dataset):
        return glob.glob(pdb_dataset)

    if _is_pdb_file(pdb_dataset, extension=extension):
        return [pdb_dataset]

    if _is_pdb_list_in_file(pdb_dataset, extension=extension):
        return _get_pdb_list_from_file(pdb_dataset, extension=extension)

    if _is_pdb_dir(pdb_dataset, extension=extension):
        return _get_pdb_dir_filenames(pdb_dataset, extension=extension)

    pdb_snapshot_filenames = _get_pdb_snapshot_filenames(
        pdb_dataset, extension=extension)
    if len(pdb_snapshot_filenames) != 0:
        return pdb_snapshot_filenames

    return []


def _is_pdb_list_in_file(pdb_file_list, extension=None):
    """Check if provided file contains a list of pdbs."""
    if not os.path.isfile(pdb_file_list):
        return False
    return len(_get_pdb_list_from_file(pdb_file_list, extension)) != 0


def _is_pdb_dir(pdb_dir, recurse=True, extension=None):
    """Check if provided file is a directory containing pdbs."""
    if not os.path.isdir(pdb_dir):
        return False
    return len(_get_pdb_dir_filenames(pdb_dir, recurse=recurse,
                                      extension=extension)) != 0


def _is_pdb_list(pdb_list):
    """Check if provided dataset is list of pdb files."""
    if isinstance(pdb_list, list) or isinstance(pdb_list, np.ndarray):
        return True
    return False


def _is_pdb_file(pdb_file, extension=None):
    """Check if provided file is a pdb file."""
    if not os.path.isfile(pdb_file):
        return False
    _, ext = os.path.splitext(pdb_file)
    if extension is None:
        return ext == ".gz" or ".pdb" in ext
    else:
        return ext == extension


def _is_pdb_regex(pdb_regex):
    """Check if provided file is a pdb regex."""
    return '*' in pdb_regex


def _is_pdb_snapshot(pdb_snapshot, extension=None):
    """Check if provided file is a directory of pdb groups."""
    return len(_get_pdb_snapshot_filenames(
        pdb_snapshot, extension=extension)) != 0


def _get_pdb_list_from_file(pdb_file_list, extension=None):
    """Get pdb files from a text file containing one pdb file per line."""
    with open(pdb_file_list, 'r') as f:
        content = f.readlines()
    candidate_files = [x.rstrip() for x in content]
    pdb_files = [x for x in candidate_files
                 if _is_pdb_file(x, extension=extension)]
    return pdb_files


def _get_pdb_dir_filenames(pdb_dir, recurse=True, extension=None):
    """Get files, given a pdb dir path."""
    pdb_filenames = []

    if recurse:
        for file in os.listdir(pdb_dir):
            new_dir = os.path.join(pdb_dir, file)
            if os.path.isdir(new_dir):
                pdb_filenames += _get_pdb_dir_filenames(new_dir, recurse=False,
                                                        extension=extension)

    if extension is None:
        pdb_filenames += glob.glob(pdb_dir + '/*.pdb*.gz') + \
            glob.glob(pdb_dir + '/*.pdb[0-9]') + \
            glob.glob(pdb_dir + '/*.pdb')
    else:
        pdb_filenames += glob.glob(pdb_dir + '/*' + extension)

    pdb_filenames = [x for x in pdb_filenames if os.path.isfile(x)]

    return pdb_filenames


def _get_pdb_snapshot_filenames(pdb_snapshot, extension=None):
    """Get files, given a pdb snapshot path."""
    pdb_groups = glob.glob(pdb_snapshot + '/[A-Za-z0-9][A-Za-z0-9]')
    valid = False
    for group in pdb_groups:
        if _is_pdb_dir(group, recurse=False, extension=extension):
            valid = True
            break

    if not valid:
        # Try and see if it stored under RCSB snapshot location.
        pdb_snapshot = pdb_snapshot + '/data/biounit/PDB/divided'
        pdb_groups = glob.glob(pdb_snapshot + '/[a-z0-9][a-z0-9]')

    pdb_filenames = []
    for pdb_group in pdb_groups:
        pdb_filenames.extend(_get_pdb_dir_filenames(pdb_group, recurse=False,
                                                    extension=extension))
    return pdb_filenames
