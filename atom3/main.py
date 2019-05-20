"""Argument processing and routing to appropriate subcommand."""
import argparse as ap
import logging
import os
import socket
import subprocess
import sys


def add_pipeline_parsers(subparsers, pp):
    """Add new pandas and tfrecords pipeline."""
    import parse as pa
    pa.add_parse_parser(subparsers, pp)

    import complex as comp
    comp.add_complexes_parser(subparsers, pp)

    import pair as pa
    pa.add_pairs_parser(subparsers, pp)

    import conservation as co
    co.add_conservation_parser(subparsers, pp)


def main():
    """Process all arguments."""
    p = ap.ArgumentParser(description='3D Atom Processing.')

    pp = ap.ArgumentParser(add_help=False)
    pp.add_argument('-l', metavar='log', type=str,
                    help='log file to output to (default: %(default)s)')

    subparsers = p.add_subparsers(title='subcommands',
                                  metavar='SUBCOMMAND   ',
                                  help='DESCRIPTION')

    # New pandas pipeline.
    add_pipeline_parsers(subparsers, pp)

    args = p.parse_args()

    root = logging.getLogger()
    map(root.removeHandler, root.handlers[:])
    if args.l is None:
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s %(process)d: ' +
                            '%(message)s',
                            level=logging.INFO)
    else:
        log_dir = os.path.dirname(args.l)
        if len(log_dir) != 0 and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(filename=args.l,
                            format='%(asctime)s %(levelname)s %(process)d: ' +
                            '%(message)s',
                            level=logging.INFO)

    logging.info('=================== CALL ===================')
    logging.info('Host is {:}'.format(socket.gethostname()))
    logging.info('Git hash is {:}'.format(
        subprocess.check_output(['git', 'rev-parse', 'HEAD'])).rstrip())
    logging.info('{}'.format(' '.join(sys.argv)))
    args.func(args)
    logging.info('================= END CALL =================')


if __name__ == "__main__":
    main()
