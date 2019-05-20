"""A full test case consisting of a bound and matching unbound complex."""
import logging
import os
import pickle
import tempfile

import Bio.Blast.Applications as app
import Bio.PDB
import Bio.SeqIO
import Bio.SVDSuperimposer as svd
import multiprocessing as mp
import numpy as np
import parallel as par

import atom3.database as db
import atom3.sequence as sequ
import atom3.structure as struct


def add_clean_parser(subparsers, pp):
    """Add parser."""

    def generate_all_clean_complexes_main(args):
        generate_all_clean_complexes(args)

    cp = subparsers.add_parser('clean', description="DB5 cleaning",
                               help='generate clean DB5 files',
                               parents=[pp])
    cp.set_defaults(func=generate_all_clean_complexes_main)
    cp.add_argument('pdb_dataset', type=str,
                    help='pdb file or dir to clean.')
    cp.add_argument('output_dir', type=str,
                    help='where to write cleaned output files.')
    cp.add_argument(
        '-s', '--style', dest='style', choices=['db5', 'dockground'],
        default='db5', help='style of dataset (default: db5)')
    cp.add_argument('-c', metavar='cpus', default=mp.cpu_count(), type=int,
                    help='number of cpus to use for processing (default:'
                    ' number processors available on current machine)')


def generate_all_clean_complexes(args):
    """Clean all complexes in input_dir, writing them out to output_dir."""
    requested_keys = get_complex_pdb_codes(args.pdb_dataset)
    produced_filenames = db.get_structures_filenames(args.output_dir)
    produced_keys = []
    for pdb_code in requested_keys:
        res = get_files_for_complex(pdb_code, produced_filenames, 'db5')
        if len([x for x in res if x is None]) == 0:
            produced_keys.append(pdb_code)
    work_keys = [key for key in requested_keys if key not in produced_keys]

    logging.info("{:} requested keys, {:} produced keys, {:} work keys"
                 .format(len(requested_keys), len(produced_keys),
                         len(work_keys)))

    inputs = [(pc, args.pdb_dataset, args.output_dir + '/' + pc, args.style)
              for pc in work_keys]
    par.submit_jobs(_generate_clean_complex, inputs, args.c)


def get_complex_pdb_codes(pdb_dataset):
    """Get complexes in provided directory."""
    complexes = set()
    for structure in db.get_structures_filenames(pdb_dataset):
        complexes.add(db.get_pdb_code(structure))
    return complexes


def get_files_for_complex(pdb_code, pdb_dataset, style):
    """Get case files corresponding to provided pdb_code."""
    lu = find_of_type(
        pdb_code, pdb_dataset, receptor=False, bound=False, style=style)
    ru = find_of_type(
        pdb_code, pdb_dataset, receptor=True, bound=False, style=style)
    if style == 'db5':
        lb = find_of_type(
            pdb_code, pdb_dataset, receptor=False, bound=True, style=style)
        rb = find_of_type(
            pdb_code, pdb_dataset, receptor=True, bound=True, style=style)
    elif style == 'dockground':
        b = find_of_type(
            pdb_code, pdb_dataset, receptor=None, bound=True, style=style)
        lb = b
        rb = b
    return lu, lb, ru, rb


def find_of_type(pdb_name_query, pdb_dataset, receptor, bound, style):
    """Get matching partner of provided file."""
    pdb_code = db.get_pdb_code(pdb_name_query)
    results = None
    for pdb_name in db.get_structures_filenames(pdb_dataset):
        if db.get_pdb_code(pdb_name) == pdb_code:
            if is_of_type(pdb_name, style, receptor=receptor, bound=bound):
                results = pdb_name
    return results


def _has_symbol(symbol, name):
    """
    Check if has provided symbol in name.

    Recognizes either the _SYMBOL pattern at end of string, or _SYMBOL_ in
    middle.
    """
    return name.endswith('_' + symbol) or ('_' + symbol + '_') in name


def is_of_type(pdb_name, style, receptor=None, bound=None):
    """Check if pdb_name is of requested type."""
    pdb_name = db.get_pdb_name(pdb_name, with_type=False)
    if receptor is None:
        if bound is None:
            return True
        elif bound:
            if style == 'db5':
                return _has_symbol('b', pdb_name)
            elif style == 'dockground':
                # Dockground only has pdb code.
                return len(pdb_name) == 4
        else:
            return _has_symbol('u', pdb_name)

    if bound is None:
        if receptor:
            return _has_symbol('r', pdb_name) or _has_symbol('2', pdb_name)
        else:
            return _has_symbol('l', pdb_name) or _has_symbol('1', pdb_name)

    if receptor and bound:
        return _has_symbol('r_b', pdb_name) or _has_symbol('2_b', pdb_name)
    elif receptor and not bound:
        return _has_symbol('r_u', pdb_name) or _has_symbol('2_u', pdb_name)
    elif not receptor and bound:
        return _has_symbol('l_b', pdb_name) or _has_symbol('1_b', pdb_name)
    else:
        return _has_symbol('l_u', pdb_name) or _has_symbol('1_u', pdb_name)

    return False


def ligand_bound(pdb_code):
    """Augment pdb code with ligand partner and bound binding notation."""
    return pdb_code + '_l_b'


def ligand_unbound(pdb_code):
    """Augment pdb code with ligand partner and unbound binding notation."""
    return pdb_code + '_l_u'


def receptor_bound(pdb_code):
    """Augment pdb code with receptor partner and bound binding notation."""
    return pdb_code + '_r_b'


def receptor_unbound(pdb_code):
    """Augment pdb code with receptor partner and unbound binding notation."""
    return pdb_code + '_r_u'


def get_partner_regex(pdb_name, bound):
    """
    Get partner regex for provided filename.

    e.g. ligand 11as_r_u_cleaned.pdb1 would give 11as_._u_cleaned.pdb1.
    """
    new_pdb_name = list(pdb_name)
    new_pdb_name[5] = '.'
    new_pdb_name[7] = 'b' if bound else 'u'
    new_pdb_name = ['^'] + new_pdb_name + ['$']
    return "".join(new_pdb_name)


def get_binder_regex(pdb_name, receptor):
    """
    Get binder regex for provided filename.

    e.g. bound 11as_r_u_cleaned.pdb1 would give 11as_r_._cleaned.pdb1.
    """
    new_pdb_name = list(pdb_name)
    new_pdb_name[7] = '.'
    new_pdb_name[5] = 'r' if receptor else 'l'
    new_pdb_name = ['^'] + new_pdb_name + ['$']
    return "".join(new_pdb_name)


def _get_binding(pdb_filename):
    """
    Get binding mode of pdb file.  x if none.

    e.g. 11as_r_u.pdb would give u
    """
    if "_u" in pdb_filename:
        return "u"
    elif "_b" in pdb_filename:
        return "b"
    else:
        return "x"


def _get_partner(pdb_filename):
    """
    Get binding mode of pdb file.  x if none.

    e.g. 11as_r_u.pdb would give u
    """
    if "_r" in pdb_filename or "_2" in pdb_filename:
        return "r"
    elif "_l" in pdb_filename or "_1" in pdb_filename:
        return "l"
    else:
        return "x"


def get_pdb_code_with_partner(pdb_filename):
    """
    Get pdb code with partner annotated.

    e.g. 11as_r_u.pdb would give 11as_r
    """
    return db.get_pdb_code(pdb_filename) + '_' + _get_partner(pdb_filename)


def get_pdb_code_with_partner_and_binding(pdb_filename):
    """
    Get pdb code with partner and binding state annotated.

    e.g. 11as_r_u.pdb would give 11as_r_u
    """
    return db.get_pdb_code(pdb_filename) + '_' + _get_partner(pdb_filename) + \
        '_' + _get_binding(pdb_filename)


def get_pdb_code_with_binding(pdb_filename):
    """
    Get pdb code with binding state annotated.

    e.g. 11as_r_u.pdb would give 11as_u
    """
    return db.get_pdb_code(pdb_filename) + '_' + _get_binding(pdb_filename)


def _tmp_fasta(seq):
    """Output provided sequences as temporaray FASTA file."""
    output = tempfile.NamedTemporaryFile().name
    sequ.write_fasta({output: seq}, output)
    return output


def _get_seq_and_atoms(filename):
    """Form dictionaries mapping from chain to their sequence and atoms."""
    seqs = {}
    all_atoms = {}
    structure = struct.parse_structure(filename)
    pdb_name = db.get_pdb_name(filename)
    for (chain, residues) in \
            struct.get_chain_to_valid_residues(structure, pdb_name):
        atoms = []
        for residue in residues:
            atoms.append(np.array(residue[['x', 'y', 'z']], dtype='f4'))
        if len(residues) != 0:
            # Ignore zero-length peptides.
            seqs[chain] = residues
            all_atoms[chain] = np.array(atoms)
    return all_atoms, seqs


def _get_alignments(seq_u, atom_u, seq_b, atom_b):
    """
    Get alignment between two chains.

    Does structural and sequence alignment.
    """
    fasta_u = _tmp_fasta(seq_u)
    fasta_b = _tmp_fasta(seq_b)

    # Only use small word size for small sequences.
    # TODO. This is a bit brittle.
    word_size = 2 if len(seq_u) < 10 and len(seq_b) < 10 else 3

    blastp_cline = app.NcbiblastpCommandline(
        subject=fasta_u, query=fasta_b,
        outfmt="'10 qstart qend sstart send qseq sseq ppos evalue'",
        num_alignments=1, word_size=word_size, culling_limit=1, evalue=0.1)
    out, err = blastp_cline()

    # The trailing token is empty.
    alignments = [x for x in out.split('\n') if x]
    b2r, u2r = {}, {}
    b2u, u2b = {}, {}
    aligned_pos_b, aligned_pos_u = [], []
    all_ppos = []
    if len(out) == 0:
        # No hits found.
        return 0.0, float('inf'), (None, None)

    warned = False
    if len(alignments) > 1:
        logging.warning("More than one alignment found.")
    for i, curr in enumerate(alignments):
        start_b, end_b, start_u, end_u, align_b, align_u, ppos, evalue = \
            curr.split(',')
        start_b, end_b = int(start_b), int(end_b)
        start_u, end_u = int(start_u), int(end_u)
#        logging.info('Alignment {:} (score {:}) from {:} to {:} on bound, '
#                     '{:} to {:} on unbound.'.format(
#                         i, evalue, start_b, end_b, start_u, end_u))
        idx_b, idx_u = start_b - 1, start_u - 1
        assert len(align_u) == len(align_b)
        align_size = len(align_u)
        for k in range(align_size):
            if align_b[k] != '-' and align_u[k] != '-':
                if idx_b not in b2u and idx_u not in u2b:
                    b2u[idx_b] = idx_u
                    u2b[idx_u] = idx_b
                    aligned_pos_b.append(idx_b)
                    aligned_pos_u.append(idx_u)
                else:
                    if not warned:
                        logging.warning('ignoring double prediction {:} bound '
                                        'to {:} unbound'.format(idx_u, idx_b))
                        logging.warning('not showing future warnings for this '
                                        'alignment')
                        warned = True
            if align_u[k] != '-':
                idx_u += 1
            if align_b[k] != '-':
                idx_b += 1
        all_ppos.append((align_size, float(ppos)))

    idx_u, idx_b = 0, 0
    idx_r = 1
    u2r, b2r = {}, {}

    while idx_u != len(seq_u) or idx_b != len(seq_b):
        if idx_u in u2b and idx_b in b2u:
            u2r[idx_u] = idx_r
            b2r[idx_b] = idx_r
            idx_u += 1
            idx_b += 1
        elif idx_u not in u2b and idx_u != len(seq_u):
            u2r[idx_u] = idx_r
            idx_u += 1
        elif idx_b not in b2u and idx_b != len(seq_b):
            b2r[idx_b] = idx_r
            idx_b += 1
        idx_r += 1

    total = 0
    total_ppos = 0
    for align_size, ppos in all_ppos:
        total_ppos += ppos * align_size
        total += align_size
    total_ppos /= total

    sup = svd.SVDSuperimposer()
    sup.set(atom_u[aligned_pos_u], atom_b[aligned_pos_b])
    return total_ppos, sup.get_init_rms(), (b2r, u2r)


def _compute_similarities(unbound, bound):
    """Get all possible alignments between chains of two PDB files."""
    atoms_u, seqs_u = _get_seq_and_atoms(unbound)
    atoms_b, seqs_b = _get_seq_and_atoms(bound)
    alignments = {}
    chains_u, chains_b = seqs_u.keys(), seqs_b.keys()
    num_u, num_b = len(seqs_u), len(seqs_b)
    seq_sims = np.zeros((num_u, num_b))
    struct_sims = np.zeros((num_u, num_b))
    for i, chain_u in enumerate(chains_u):
        for j, chain_b in enumerate(chains_b):
            # logging.info("unbound chain {:} vs. bound chain {:}"
            #              .format(chain_u, chain_b))
            seq_sims[i, j], struct_sims[i, j], alignments[(i, j)] = \
                _get_alignments(seqs_u[chain_u], atoms_u[chain_u],
                                seqs_b[chain_b], atoms_b[chain_b])
    return chains_b, chains_u, seq_sims, struct_sims, alignments


def _get_chain_mapping(unbound, bound, style):
    """Based on alignments, get mapping from chain to chain and res to res."""
    chains_b, chains_u, seq_sims, struct_sims, alignments = \
        _compute_similarities(unbound, bound)

    b2u_chain = {}
    u2b_chain = {}
    b2r_res = {}
    u2r_res = {}
    for j, chain_b in enumerate(chains_b):
        best_chain = None
        best_seq_score = 0
        best_struct_score = float('inf')
        best_alignment = None
        for i, chain_u in enumerate(chains_u):
            curr_seq_score = seq_sims[i, j]
            curr_struct_score = struct_sims[i, j]
            if best_chain is None and curr_seq_score != 0.0:
                best_chain = chain_u
                best_seq_score = curr_seq_score
                best_struct_score = curr_struct_score
                best_alignment = alignments[(i, j)]
            elif curr_seq_score > best_seq_score - 20 and \
                    curr_struct_score < best_struct_score:
                # If the sequence similarity score is comparable and structural
                # distance is better.
                best_chain = chain_u
                best_seq_score = curr_seq_score
                best_struct_score = curr_struct_score
                best_alignment = alignments[(i, j)]
        if best_chain is not None:
            # For DB5, we assume that each bound chain has a matching unbound,
            # but not the reverse.
            b2u_chain[chain_b] = best_chain
            u2b_chain[best_chain] = chain_b
            b2r_res[chain_b] = best_alignment[0]
            u2r_res[best_chain] = best_alignment[1]

    best_seq_match_for_each_bound = np.max(seq_sims, axis=0)
    # If the sequence similarity score is comparable, we rely on structural
    # distance.
    seq_threshold_for_each_bound = best_seq_match_for_each_bound - 20
    candidates_for_each_bound = seq_sims > seq_threshold_for_each_bound

    best_seq_match_for_each_unbound = np.max(seq_sims, axis=1)
    # If the sequence similarity score is comparable, we rely on structural
    # distance.
    seq_threshold_for_each_unbound = best_seq_match_for_each_unbound - 20
    candidates_for_each_unbound = seq_sims.T > seq_threshold_for_each_unbound

    b2u_proposed = {}
    u2b_proposed = {}
    for i, bound_candidates in enumerate(candidates_for_each_unbound.T):
        if np.max(seq_sims[i, :]) == 0.0:
            continue
        # All bound sorted by how structurally similar they are to unbound.
        struct_idx = np.argsort(struct_sims[i, :])
        # Choose best bound by finding best structure that has OK sequence
        # match.
        best_bound = struct_idx[np.argmax(bound_candidates[struct_idx])]
        u2b_proposed[i] = best_bound
    for i, unbound_candidates in enumerate(candidates_for_each_bound.T):
        if np.max(seq_sims[:, i]) == 0.0:
            continue
        # All unbound sorted by how structurally similar they are to bound.
        struct_idx = np.argsort(struct_sims[:, i])
        # Choose best bound by finding best structure that has OK sequence
        # match.
        best_unbound = struct_idx[np.argmax(unbound_candidates[struct_idx])]
        b2u_proposed[i] = best_unbound

    b2u_chain_proposed = {}
    for b, u in b2u_proposed.items():
        b2u_chain_proposed[chains_b[b]] = chains_u[u]
    u2b_chain_proposed = {}
    for u, b in u2b_proposed.items():
        u2b_chain_proposed[chains_u[u]] = chains_b[b]

    if style == 'db5':
        # For DB5, we assume that each bound chain has a matching unbound, but
        # not the reverse.
        assert u2b_chain_proposed == u2b_chain
        assert b2u_chain_proposed == b2u_chain
        u2b_chain, b2u_chain = {}, {}
        for b, u in b2u_proposed.items():
            chain_b, chain_u = chains_b[b], chains_u[u]
            b2u_chain[chain_b] = chain_u
            u2b_chain[chain_u] = chain_b
            b2r_res[chain_b] = alignments[u, b][0]
            u2r_res[chain_u] = alignments[u, b][1]
    elif style == 'dockground':
        # For dockground, we assume that each unbound chain has a matching
        # bound, but not the reverse.
        u2b_chain, b2u_chain = {}, {}
        for u, b in u2b_proposed.items():
            chain_b, chain_u = chains_b[b], chains_u[u]
            b2u_chain[chain_b] = chain_u
            u2b_chain[chain_u] = chain_b
            b2r_res[chain_b] = alignments[u, b][0]
            u2r_res[chain_u] = alignments[u, b][1]

    assert len(b2u_chain) == len(u2b_chain)

    # Map bound and unbound chains to common reference.
    b2r_chain = {}
    u2r_chain = {}
    reference_chain = 'A'
    for chain in b2u_chain:
        matching_chain = b2u_chain[chain]
        b2r_chain[chain] = reference_chain
        u2r_chain[matching_chain] = reference_chain
        reference_chain = chr(ord(reference_chain) + 1)

    for unmatched_u in [cu for cu in chains_u if cu not in u2b_chain]:
        logging.warning("{:} chain {:} has no match found!"
                        .format(
                            get_pdb_code_with_partner_and_binding(unbound),
                            unmatched_u))
        u2r_chain[unmatched_u] = reference_chain
        reference_chain = chr(ord(reference_chain) + 1)
    for unmatched_b in [cb for cb in chains_b if cb not in b2u_chain]:
        if style == 'dockground':
            # We don't care about unmatched bound in dockground.
            continue
        logging.warning("{:} chain {:} has no match found!"
                        .format(
                            get_pdb_code_with_partner_and_binding(bound),
                            unmatched_b))
        b2r_chain[unmatched_b] = reference_chain
        reference_chain = chr(ord(reference_chain) + 1)

    return b2r_chain, u2r_chain, b2r_res, u2r_res


def _generate_clean_unbound_bound(filename_u, filename_b, results_dir, style):
    """Perform alignment on unbound and bound files to standardize them."""
    b2r_chain, u2r_chain, b2r_res, u2r_res = \
        _get_chain_mapping(filename_u, filename_b, style)

    aug_pdb_code_u = get_pdb_code_with_partner_and_binding(filename_u)
    if style == 'db5':
        aug_pdb_code_b = get_pdb_code_with_partner_and_binding(filename_b)
        pdb_extension = 'pdb'
    elif style == 'dockground':
        partner = _get_partner(filename_u)
        aug_pdb_code_b = db.get_pdb_code(filename_b) + '_' + partner + '_b'
        pdb_extension = db.get_pdb_type(filename_b)

    output_filename_u = results_dir + '/' + aug_pdb_code_u + '_cleaned.' + \
        pdb_extension
    output_filename_b = results_dir + '/' + aug_pdb_code_b + '_cleaned.' + \
        pdb_extension
    _generate_reference(
        filename_b, b2r_chain, b2r_res, output_filename_b, style)
    _generate_reference(
        filename_u, u2r_chain, u2r_res, output_filename_u, style)

    output_mapping_u = results_dir + '/' + aug_pdb_code_u + '_toref.pkl'
    output_mapping_b = results_dir + '/' + aug_pdb_code_b + '_toref.pkl'
    _generate_mapping(u2r_chain, u2r_res, output_mapping_u)
    _generate_mapping(b2r_chain, b2r_res, output_mapping_b)


def _generate_mapping(s2r_chain, s2r_res, output_filename):
    """Write structure-to-reference chain and residue mappings."""
    mappings = {}
    mappings['chain'] = s2r_chain
    mappings['res'] = s2r_res
    with open(output_filename, 'wb') as f:
        pickle.dump(mappings, f)


def _generate_reference(pdb_filename, s2r_chain, s2r_res, output_filename,
                        style):
    """Transform PDB structure to a reference structure."""
    biopy_structure = db.parse_biopython_structure(pdb_filename)
    pdb_name = db.get_pdb_name(pdb_filename)

    new_model = Bio.PDB.Model.Model('0')
    new_structure = Bio.PDB.Structure.Structure('')
    for (chain, residues) in \
            struct.get_chain_to_valid_residues(biopy_structure, pdb_name):
        if style == 'dockground' and chain not in s2r_chain:
            # If we are in dockground, we allow ourselves to remove unmapped
            # chains.
            continue
        ref_chain = s2r_chain[chain]

        if chain in s2r_res:
            # If we have an alignment for this chain.
            new_chain = Bio.PDB.Chain.Chain(ref_chain)
            for i, residue in enumerate(residues):
                if residue.id[0] != ' ':
                    continue
                residue.segid = ""
                residue.id = (' ', s2r_res[chain][i], residue.id[2])
                new_chain.add(residue)
        else:
            # Else, just remove segment ID.
            new_chain = Bio.PDB.Chain.Chain(ref_chain)
            for i, residue in enumerate(residues):
                residue.segid = ""
        new_model.add(new_chain)

    new_structure.add(new_model)
    w = Bio.PDB.PDBIO()
    w.set_structure(new_structure)
    w.save(output_filename)


def _generate_clean_complex(pdb_code, containing_dir, results_dir, style):
    """Clean up provided complex, sending it to output_dir."""
    logging.info(pdb_code)
    lu, lb, ru, rb = get_files_for_complex(pdb_code, containing_dir, style)

    if lu is None or ru is None or lb is None or rb is None:
        logging.warning("Skipping {:} since not all pdb files present."
                        .format(pdb_code))
        return

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

#    logging.info("Receptor")
    _generate_clean_unbound_bound(ru, rb, results_dir, style)
#    logging.info("Ligand")
    _generate_clean_unbound_bound(lu, lb, results_dir, style)
