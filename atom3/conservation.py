import cPickle as pickle
import logging
import multiprocessing as mp
import os
import subprocess
import timeit

import pandas as pd
import parallel as par

import atom3.database as db
import atom3.sequence as sequ


def add_conservation_parser(subparsers, pp):
    """Add parser."""

    def map_all_pssms_main(args):
        map_all_pssms(args.pkl_dataset, args.blastdb, args.output_dir, args.c)

    ap = subparsers.add_parser(
        'conservation', description='sequence conservation',
        help='compute sequence conservation features',
        parents=[pp])
    ap.set_defaults(func=map_all_pssms_main)
    ap.add_argument('pkl_dataset', metavar='pkl', type=str,
                    help='parsed dataset')
    ap.add_argument('blastdb', metavar='bdb', type=str,
                    help='blast database to do lookups on')
    ap.add_argument('output_dir', metavar='output', type=str,
                    help='directory to output to')
    ap.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                    ' number processors available on current machine)')


def gen_pssm(pdb_filename, blastdb, output_filename):
    """Generate PSSM and PSFM from sequence."""
    pdb_name = db.get_pdb_name(pdb_filename)
    out_dir = os.path.dirname(output_filename)
    work_dir = os.path.join(out_dir, 'work')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_format = work_dir + "/{:}.fa"
    id_format = work_dir + "/{:}.cpkl"
    chains, chain_fasta_filenames, id_filenames = sequ.pdb_to_fasta(
        pdb_filename, fasta_format, id_format, True)

    pssms = []
    for chain, chain_fasta_filename, id_filename in \
            zip(chains, chain_fasta_filenames, id_filenames):
        basename = os.path.splitext(chain_fasta_filename)[0]
        pssm_filename = "{}.pssm".format(basename)
        blast_filename = "{}.blast".format(basename)
        clustal_filename = "{}.clustal".format(basename)
        al2co_filename = "{}.al2co".format(basename)
        if not os.path.exists(pssm_filename):
            logging.info("Blasting {:}".format(chain_fasta_filename))
            _blast(chain_fasta_filename, pssm_filename, blast_filename,
                   blastdb)

        if not os.path.exists(pssm_filename):
            logging.warning("No hits for {:}".format(chain_fasta_filename))
            # Create empty file.
            open(pssm_filename, 'w').close()

        if not os.path.exists(clustal_filename):
            logging.info("Converting {:}".format(blast_filename))
            _to_clustal(blast_filename, clustal_filename)

        if not os.path.exists(al2co_filename):
            logging.info("Al2co {:}".format(al2co_filename))
            _al2co(clustal_filename, al2co_filename)

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
            psfm.columns = pssm.columns
            del psfm['orig']

            # Combine both into one.
            psfm = psfm.add_prefix('psfm_')
            pssm = pssm.add_prefix('pssm_')
            pssm.rename(columns={'pssm_orig': 'resname'}, inplace=True)
            al2co = pd.read_table(
                al2co_filename, delim_whitespace=True, usecols=[2],
                names=['al2co'])
            pssm = pd.concat([pssm, psfm, al2co], axis=1)

        else:
            logging.warning("No pssm found for {:} (model {:}, chain {:})"
                            .format(pdb_name, chain[-2], chain[-1]))
            pssm, psfm = None, None

        pdb_name = db.get_pdb_name(pdb_filename)
        key = pdb_name + '-' + chain[-2] + '-' + chain[-1]
        pos_to_res = pickle.load(open(id_filename))[key]

        pssm['pdb_name'] = db.get_pdb_name(pdb_filename)
        pssm['model'] = chain[0]
        pssm['chain'] = chain[1]
        pssm['residue'] = pos_to_res
        pssms.append(pssm)
    pssms = pd.concat(pssms)
    return pssms


def map_pssms(pdb_filename, blastdb, output_filename):
    pdb_name = db.get_pdb_name(pdb_filename)
    start_time = timeit.default_timer()
    start_time_blasting = timeit.default_timer()
    pis = gen_pssm(pdb_filename, blastdb, output_filename)
    num_chains = len(pis.groupby(['pdb_name', 'model', 'chain']))
    elapsed_blasting = timeit.default_timer() - start_time_blasting

    parsed = pd.read_pickle(pdb_filename)
    parsed = parsed.merge(
        pis, on=['model', 'pdb_name', 'chain', 'residue', 'resname'])

    start_time_writing = timeit.default_timer()
    parsed.to_pickle(output_filename)
    elapsed_writing = timeit.default_timer() - start_time_writing

    elapsed = timeit.default_timer() - start_time
    logging.info(
        ('For {:d} pssms generated from {} spent {:05.2f} blasting, '
         '{:05.2f} writing, and {:05.2f} overall.')
        .format(
             num_chains,
             pdb_name,
             elapsed_blasting,
             elapsed_writing,
             elapsed))


def _blast(query, output_pssm, output, blastdb):
    """Run PSIBlast on specified input."""
    psiblast_command = "psiblast -db {:} -query {:} -out_ascii_pssm {:} " + \
        "-save_pssm_after_last_round -out {:}"
    log_out = "{}.out".format(output)
    log_err = "{}.err".format(output)
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = psiblast_command.format(
                blastdb, query, output_pssm, output)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(
                command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def _to_clustal(psiblast_in, clustal_out):
    """Convert PSIBlast output to CLUSTAL format."""
    log_out = "{}.out".format(clustal_out)
    log_err = "{}.err".format(clustal_out)
    mview_command = "mview -in blast -out clustal {:} | tail -n+4 > {:}"
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = mview_command.format(psiblast_in, clustal_out)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(
                command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def _al2co(clustal_in, al2co_out):
    """Use al2co on CLUSTAL format."""
    log_out = "{}.out".format(al2co_out)
    log_err = "{}.err".format(al2co_out)
    al2co_command = "al2co -i {:} -g 0.9 | head -n -12 > {:}"
    with open(log_out, 'a') as f_out:
        with open(log_err, 'a') as f_err:
            command = al2co_command.format(clustal_in, al2co_out)
            f_out.write('=================== CALL ===================\n')
            f_out.write(command + '\n')
            subprocess.check_call(
                command, shell=True, stderr=f_err, stdout=f_out)
            f_out.write('================= END CALL =================\n')


def map_all_pssms(pdb_dataset, blastdb, output_dir, num_cpus):
    ext = '.pkl'
    requested_filenames = \
        db.get_structures_filenames(pdb_dataset, extension=ext)
    requested_keys = [db.get_pdb_name(x)
                      for x in requested_filenames]
    produced_filenames = db.get_structures_filenames(
        output_dir, extension='.pkl')
    produced_keys = [db.get_pdb_name(x)
                     for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    work_filenames = [x[0] for x in
                      db.get_all_filenames(
                          work_keys, pdb_dataset, extension=ext,
                          keyer=lambda x: db.get_pdb_name(x))]

    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + db.get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        output_filenames.append(
            sub_dir + '/' + db.get_pdb_name(pdb_filename) + ".pkl")

    logging.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))
    inputs = [(key, blastdb, output)
              for key, output in zip(work_filenames, output_filenames)]
    par.submit_jobs(map_pssms, inputs, num_cpus)
