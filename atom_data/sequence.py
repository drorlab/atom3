"""Code to do with pdb structure sequences."""
import collections as col
import cPickle as pickle
import logging
import multiprocessing as mp
import os
import subprocess
import timeit

import Bio
import Bio.PDB.Polypeptide as poly
import h5py
import numpy as np
import pandas as pd

import database as db
import structure as struct
import parallel as par

PSSMInfo = col.namedtuple('PSSMInfo', ['pdb_filename', 'chain', 'pssm', 'psfm',
                                       'id_filename', 'pssm_filename', 'key'])

sem = mp.Semaphore()


def add_blastdb_parser(subparsers, pp):
    """Add parser."""

    def write_pdb_dataset_to_blast_db_main(args):
        write_pdb_dataset_to_blast_db(args.pdb, args.blast_db, args.d)

    bp = subparsers.add_parser(
        'makeblastdb', description="create blast db",
        help='make a file containing protein sequences', parents=[pp])
    bp.add_argument('pdb', type=str,
                    help='pdb file or dir to make blast db from.')
    bp.add_argument('blast_db', type=str, help='location to write db.')
    bp.add_argument('-d', action='store_true',
                    help='extract more detailed information from '
                    'protprep-ed proteins')
    bp.set_defaults(func=write_pdb_dataset_to_blast_db_main)


def write_pdb_dataset_to_blast_db(pdb_dataset, blast_db, detailed):
    """Write provided pdb dataset to blast db, for use with BLAST."""
    if detailed:
        filenames = db.get_structures_filenames(pdb_dataset, extension='.pkl')
    else:
        filenames = db.get_structures_filenames(pdb_dataset)
    if len(filenames) == 0:
        logging.warning("No pdb files detected to make database from.")
        return
    logging.info("Forming database from {:} structure".format(len(filenames)))

    flat_map = {}
    for pdb_filename in filenames:
        pdb_name = db.get_pdb_name(pdb_filename)
        logging.info("Processing {:}".format(pdb_name))
        structure = struct.parse_structure(pdb_filename)
        for (chain, residues) in struct.get_chain_to_valid_residues(structure,
                                                                    pdb_name):
            flat_map['-'.join(chain)] = residues

    if len(flat_map) == 0:
        logging.warning("No valid residues detected in database.")
        return

    logging.info("Writing fasta db")
    write_fasta(flat_map, blast_db)

    logging.info("Forming blast db")
    subprocess.check_output('makeblastdb -in ' + blast_db + ' -dbtype prot',
                            shell=True)


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
    pdb_name = db.get_pdb_name(pdb_filename, with_type=False)
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


def _blast(query, output, blastdb, log_out, log_err):
    """Run PSIBlast on specified input."""
    psiblast_command = "psiblast -db {:} -query {:} -out_ascii_pssm {:} " + \
        "-save_pssm_after_last_round"
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = psiblast_command.format(blastdb, query, output)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(
                command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def gen_pssm(pdb_filename, blastdb, out_dir):
    """Generate PSSM and PSFM from sequence."""
    pdb_name = db.get_pdb_name(pdb_filename)
    fasta_format = out_dir + "/{:}.fa"
    id_format = out_dir + "/{:}.pkl"
    chains, chain_fasta_filenames, id_filenames = pdb_to_fasta(
        pdb_filename, fasta_format, id_format, True)

    pssms = []
    for chain, chain_fasta_filename, id_filename in \
            zip(chains, chain_fasta_filenames, id_filenames):
        basename = os.path.splitext(chain_fasta_filename)[0]
        pssm_filename = "{}.pssm".format(basename)
        log_out = "{}.out".format(basename)
        log_err = "{}.err".format(basename)
        if not os.path.exists(pssm_filename):
            logging.info("Blasting {:}".format(chain_fasta_filename))
            _blast(chain_fasta_filename, pssm_filename, blastdb, log_out,
                   log_err)
        if not os.path.exists(pssm_filename):
            logging.warning("No hits for {:}".format(chain_fasta_filename))
            # Create empty file.
            open(pssm_filename, 'w').close()

        if os.stat(pssm_filename).st_size != 0:
            pssm = pd.read_table(
                pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
                engine='python', usecols=range(20), index_col=[0, 1])
            pssm = pssm.reset_index()
            del pssm['level_0']
            pssm.rename(columns={'level_1': 'orig'}, inplace=True)

            pscm = pd.read_table(
                pssm_filename, skiprows=2, skipfooter=6, delim_whitespace=True,
                engine='python', usecols=range(20, 40), index_col=[0, 1])
            psfm = pscm.applymap(lambda x: x / 100.)
            psfm = psfm.reset_index()
            del psfm['level_0']
            psfm.rename(columns={'level_1': 'orig'}, inplace=True)
            psfm.columns = pssm.columns
        else:
            logging.warning("No pssm found for {:} (model {:}, chain {:})"
                            .format(pdb_name, chain[-2], chain[-1]))
            pssm, psfm = None, None

        key = pdb_name + '-' + chain[-2] + '-' + chain[-1]
        pi = PSSMInfo(
            pdb_filename, chain, pssm, psfm, id_filename, pssm_filename, key)
        pssms.append(pi)
    return pssms


def add_mappssms_parser(subparsers, pp):
    """Add parser."""

    def map_all_pssms_main(args):
        map_all_pssms(args.pdb_dataset, args.blastdb, args.pssm_hdf5,
                      args.tmp_dir, args.c)

    ap = subparsers.add_parser(
        'mappssms', description="pssm generation",
        help='generate and map pssms to file',
        parents=[pp])
    ap.set_defaults(func=map_all_pssms_main)
    ap.add_argument('pdb_dataset', type=str,
                    help='pdb dataset to divide up.')
    ap.add_argument('blastdb', type=str,
                    help='blastdb to check sequence against')
    ap.add_argument('pssm_hdf5', type=str,
                    help='hdf5 file to output pssms to')
    ap.add_argument('tmp_dir', type=str,
                    help='directory to send temporary information to')
    ap.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                    ' number processors available on current machine)')


def map_all_pssms(pdb_dataset, blastdb, pssm_hdf5, tmp_dir, num_cpus):
    ext = '.pkl'
    pdb_filenames = db.get_structures_filenames(pdb_dataset, extension=ext)
    requested_keys = [db.get_pdb_name(x, with_type=False)
                      for x in pdb_filenames]
    if os.path.exists(pssm_hdf5):
        with h5py.File(pssm_hdf5, 'r') as f:
            produced_keys = f.keys()
    else:
        produced_keys = []
    work_keys = [key for key in requested_keys if key not in produced_keys]
    work_filenames = [x[0] for x in
                      db.get_all_filenames(
                          work_keys, pdb_dataset, extension=ext,
                          keyer=lambda x: db.get_pdb_name(x, with_type=False))]

    logging.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))
    inputs = [(file, blastdb, pssm_hdf5, tmp_dir)
              for file in work_filenames]
    par.submit_jobs(map_pssms, inputs, num_cpus)


def map_pssms(pdb_filename, blastdb, pssm_hdf5, tmp_dir):
    pdb_name = db.get_pdb_name(pdb_filename, with_type=False)
    start_time = timeit.default_timer()
    start_time_blasting = timeit.default_timer()
    pis = gen_pssm(pdb_filename, blastdb, tmp_dir)
    num_chains = len(pis)
    elapsed_blasting = timeit.default_timer() - start_time_blasting

    start_time_processing = timeit.default_timer()
    pssms = []
    psfms = []
    pos_to_ress = []
    for pi in pis:
        key = pdb_name + '-' + pi.chain[-2] + '-' + pi.chain[-1]
        pos_to_res = pickle.load(open(pi.id_filename))[key]
        if pi.pssm is None:
            # If weren't able to get PSSM, just make empty one.
            pssm = np.zeros((0, 20), dtype='i4')
            psfm = np.zeros((0, 20), dtype='f4')
        else:
            num_aa = pi.pssm.shape[0]
            pssm = np.zeros((num_aa, 20), dtype='i4')
            psfm = np.zeros((num_aa, 20), dtype='f4')
            for i in range(num_aa):
                pssm[i] = pi.pssm.iloc[i][1:21]
                psfm[i] = pi.psfm.iloc[i][1:21]
        pssms.append(pssm)
        psfms.append(psfm)
        pos_to_ress.append(pos_to_res)
    elapsed_processing = timeit.default_timer() - start_time_processing

    start_time_writing = timeit.default_timer()
    with sem:
        with h5py.File(pssm_hdf5, 'a') as f:
            grp = f.require_group(pdb_name)
            for i, pi in enumerate(pis):
                sgrp = grp.require_group('_'.join(pi.chain))
                sgrp.create_dataset('psfm', data=psfms[i])
                sgrp.create_dataset('pssm', data=pssms[i])
                sgrp.create_dataset('pos_to_res', data=pos_to_ress[i])
    elapsed_writing = timeit.default_timer() - start_time_writing

    elapsed = timeit.default_timer() - start_time
    logging.info(
        ('For {:d} pssms generated from {} spent {:05.2f} blasting, '
         '{:05.2f} processing, {:05.2f} writing, and {:05.2f} overall.')
        .format(
             num_chains,
             pdb_name,
             elapsed_blasting,
             elapsed_processing,
             elapsed_writing,
             elapsed))


def add_seq_information(dataset, seq_src, seqmodel_params):
    """Add sequence information to existing tf dataset."""
    import tensorflow as tf
    data = {}
    with h5py.File(seq_src, 'r') as f:
        for group in f.keys():
            data[group] = {}
            for chain in f[group].keys():
                res_to_pos = dict(
                    [(x, i) for i, x in
                     enumerate(f[group][chain]['pos_to_res'][:])])
                data[group][chain] = {
                    'pssm': f[group][chain]['pssm'][:],
                    'psfm': f[group][chain]['psfm'][:],
                    'res_to_pos': res_to_pos}

    def _get_pssm_window(pdb_names, models, chains, residues):
        num_ex = pdb_names.shape[0]
#        pdb_names = np.array(
#            [['1FQJ_l_b_cleaned', '1FQJ_l_b_cleaned']] * num_ex)
#        models = np.array([['0', '0']] * num_ex)
#        chains = np.array([['A', 'A']] * num_ex)
#        residues = np.array([['4 ', '4 ']] * num_ex)
        radius = seqmodel_params['cons_window_radius']
        size = radius * 2 + 1
        pssms = np.zeros((num_ex, 2, size, 20), dtype='i4')
        psfms = np.zeros((num_ex, 2, size, 20), dtype='f4')
        for ex in range(num_ex):
            for which in (0, 1):
                pdb_name = db.get_pdb_name(
                    pdb_names[ex, which], with_type=False)
                model = models[ex, which]
                chain = chains[ex, which]
                residue = residues[ex, which]
                if pdb_name not in data:
#                    logging.warning('{:} not present'.format(pdb_name))
                    continue
                grp = data[pdb_name][model + '_' + chain]
                if grp['pssm'].shape[0] == 0:
#                    logging.warning('{:} has no pssm'.format(pdb_name))
                    continue
                pos = grp['res_to_pos'][residue]
                start, end = pos - radius, pos + radius + 1
                # Get window around location of interest.
                for i, idx in enumerate(range(start, end)):
                    if idx >= 0 and idx < grp['pssm'].shape[0]:
                        pssms[ex, which, i] = grp['pssm'][idx]
                        psfms[ex, which, i] = grp['psfm'][idx]
        return pssms, psfms

    seq_dataset = dataset.map(
        lambda x: tf.py_func(
            _get_pssm_window,
            [x['pdb_name'], x['model'], x['chain'], x['residue']],
            [tf.int32, tf.float32]))
    dataset = tf.data.Dataset.zip((dataset, seq_dataset))
    dataset = dataset.map(
        lambda ex, seq: dict({'pssm': seq[0], 'psfm': seq[1]}, **ex))
    return dataset
