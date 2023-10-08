from argparse import ArgumentParser, MetavarTypeHelpFormatter

from global_settings import get_context
from run import debug, solve

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="·OH energy solver",
        usage="VQE for calculating ground state of ·OH.",
        formatter_class=MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--noise",
        type=str,
        default="cairo",
        choices=["cairo", "kolkata", "montreal"],
        help="Noise model. Defaults to cairo.",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="imfil",
        choices=["imfil", "snobfit", "nomad", "bobyqa", "orbit"],
        help="Optimizer for VQE. Defaults to imfil.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=500,
        help="Budget for iterations. Defaults to 500.",
    )
    parser.add_argument(
        "--bounds-shift",
        type=float,
        default=[0.25, 0.25],
        nargs=2,
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--readout-mitigator",
        type=str,
        default="local",
        choices=["local", "correlated"],
        help="""Class for readout error experiment in Qiskit.
        Defaults to local.""",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./result",
        help='Direction for saving results. Defaults to "./result".',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=170,
        choices=[20, 21, 30, 33, 36, 42, 43, 55, 67, 170],
        help="""Seed for algorithmic, transpiling, and measurement in Qiskit.
        Defaults to 170.""",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="""Shots for circuit execution. Defaults to 1000.""",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="""Threshold to divide Pauli strings.
        The absolute value of coefficients which is larger than the threshold
        can be considered as a large weight. Defaults to 1.0.""",
    )
    parser.add_argument(
        "--zne_fold", type=str, default="global", choices=["global", "random"]
    )
    parser.add_argument(
        "--zne_scale",
        type=float,
        default=[1.0, 2.0, 3.0],
        nargs="+",
        help="Scale factor for ZNE.",
    )

    args = parser.parse_args()
    print(args)
    (debug if args.debug else solve)(get_context(args))
