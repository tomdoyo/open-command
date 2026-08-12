import argparse

def common_parse_args(program_name: str) -> argparse.Namespace:
    """ Parse the options common to all open-command commands. """
    parser = argparse.ArgumentParser(
                    prog=program_name)
    parser.add_argument('year', type=int, nargs='?', default=2026, help="Specify the year to analyze. (Default: 2026)")
    return parser.parse_args()
