import collections as col
import logging
import os
import dill

import atom3.case as ca
import atom3.database as db

Complex = col.namedtuple(
    'Complex', ['name', 'bound_filenames', 'unbound_filenames'])


def add_complexes_parser(subparsers, pp):
    """Add parser."""

    def complexes_main(args):
        complexes(args.pkl_dataset, args.output_dill, args.type)

    ap = subparsers.add_parser(
        'complex', description='pickled file mapping to complexes',
        help='maps protein complexes to their individual subunits',
        parents=[pp])
    ap.set_defaults(func=complexes_main)
    ap.add_argument('pkl_dataset', type=str, help='pkl dataset')
    ap.add_argument('output_dill', metavar='complexes.dill', type=str,
                    help='dill file to output complexes to')
    ap.add_argument(
        '-t', '--type', dest='type',
        choices=['rcsb', 'db5', 'dockground', 'db5mut', 'hotspot'],
        default='rcsb', help='type of dataset (default: rcsb)')


def complexes(pkl_dataset, output_dill, type):
    if os.path.exists(output_dill):
        logging.warning("Complex file {:} already exists!".format(output_dill))
        return
    logging.info("Getting filenames...")
    filenames = db.get_structures_filenames(pkl_dataset, extension='.pkl')
    logging.info("Getting complexes...")
    complexes = get_complexes(filenames, type)
    write_complexes(complexes, output_dill)


def get_complexes(filenames, type):
    """
    Return mapping to files from same complex.

    If unbound true, will return a pair of list of files.  First one is bound
    files, second is their unbound counterpart.
    """
    if type == 'rcsb':
        raw_complexes = _get_rcsb_complexes(filenames)
    elif type == 'db5':
        raw_complexes = _get_db5_complexes(filenames)
    elif type == 'db5mut':
        raw_complexes = _get_db5_complexes_mut(filenames)
    elif type == 'hotspot':
        raw_complexes = _get_db5_complexes_hotspot_mut(filenames)
    elif type == 'dockground':
        raw_complexes = _get_db5_complexes(
            filenames, keyer=db.get_pdb_basename)
    else:
        raise RuntimeError("Unrecognized dataset type {:}".format(type))
    complexes = {
        'data': raw_complexes,
        'type': type
    }
    return complexes


def read_complexes(input_dill):
    with open(input_dill, 'r') as f:
        complexes = dill.load(f)
    return complexes


def write_complexes(complexes, output_dill):
    if os.path.exists(output_dill):
        raise RuntimeError(
            "Complex file {:} already exists!".format(output_dill))
    if not os.path.exists(os.path.dirname(output_dill)):
        os.makedirs(os.path.dirname(output_dill))
    with open(output_dill, 'w') as f:
        dill.dump(complexes, f)


def _get_rcsb_complexes(filenames):
    """Get complexes for RCSB type dataset."""
    complexes = {}
    for filename in filenames:
        name = db.get_pdb_name(filename)
        complexes[name] = Complex(name=name, bound_filenames=[filename],
                                  unbound_filenames=[])
    return complexes


def _get_db5_complexes_mut(filenames):
    filenames = [f for f in filenames if 'mut' in f] #screen the original non mutant
    basenames = [os.path.basename(f) for f in filenames]
    new_pdbs = [b[:4] + '_' + b[b.rfind('_') + 1: b.find('.')] for b in basenames]
    complexes = {}
    for pdb in new_pdbs:
        code4 = pdb[:4]
        mutnum = 'mut_' + pdb[pdb.find('_') + 1: ] + '.'
        lb = [x for x in filenames if code4 in x and mutnum in x and '_l_b_' in x][0]
        rb = [x for x in filenames if code4 in x and mutnum in x and '_r_b_' in x][0]
        lu = [x for x in filenames if code4 in x and mutnum in x and '_l_u_' in x][0]
        ru = [x for x in filenames if code4 in x and mutnum in x and '_r_u_' in x][0]
        complexes[pdb] = Complex(name=pdb, bound_filenames=[lb, rb], unbound_filenames=[lu, ru])
    return complexes


def _get_db5_complexes_hotspot_mut(filenames):
    filenames = sorted(filenames)
    complexes = {}
    file_pairs = {}
    for f in filenames:
        pdb = ''
        if 'mut' in f:
            pdb = os.path.basename(f)[:4] + f[f.rfind('_') + 1: f.find('.')]
        else:
            pdb = os.path.basename(f)[:4]
        if pdb in file_pairs:
            file_pairs[pdb] += [f]
        else:
            file_pairs[pdb] = [f]
    for pdb in file_pairs:
        if '_l_' in file_pairs[pdb][0]:
            complexes[pdb] = Complex(name=pdb, bound_filenames=file_pairs[pdb], unbound_filenames=[None, None])
        else:
            complexes[pdb] = Complex(name=pdb, bound_filenames=[file_pairs[pdb][1], file_pairs[pdb][0]], unbound_filenames=[None, None])
    return complexes


def _get_db5_complexes(filenames, keyer=db.get_pdb_code):
    """Get complexes for docking benchmark 5 type dataset."""
    pdb_codes = ca.get_complex_pdb_codes(filenames)
    complexes = {}
    for pdb_code in pdb_codes:
        lb = ca.find_of_type(
            pdb_code, filenames, receptor=False, bound=True, style='db5')
        rb = ca.find_of_type(
            pdb_code, filenames, receptor=True, bound=True, style='db5')
        lu = ca.find_of_type(
            pdb_code, filenames, receptor=False, bound=False, style='db5')
        ru = ca.find_of_type(
            pdb_code, filenames, receptor=True, bound=False, style='db5')
        if lb is None or rb is None:
            logging.warning("Skipping {:} since not all bound files present."
                            .format(pdb_code))
            continue
        if lu is None or ru is None:
            logging.warning(
                "Skipping {:} since not all unbound files present."
                .format(pdb_code))
            continue
        if keyer(lu) != keyer(lb) or keyer(lu) != keyer(rb) or \
                keyer(lu) != keyer(ru):
            logging.warning(
                "Skipping {:} since not all keys match."
                .format(pdb_code))
            continue

        complexes[keyer(lu)] = Complex(
            name=keyer(lu), bound_filenames=[lb, rb],
            unbound_filenames=[lu, ru])
    return complexes
