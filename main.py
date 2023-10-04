from argparse import ArgumentParser

from global_settings import get_context

if __name__ == "__main__":
    parser = ArgumentParser(usage="OH Energy Solver")

    parser.add_argument(
        "-n",
        "--noise",
        type=str,
        default="cairo",
        choices=["cairo", "kolkata", "montreal"],
        help="Noise model.",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="imfil",
        choices=["imfil", "snobfit", "nomad", "bobyqa", "orbit"],
        help="Optimizer for VQE.",
    )
    parser.add_argument(
        "--budget", type=int, default=500, help="Budget for iterations."
    )
    parser.add_argument(
        "--readout-mitigator",
        type=str,
        default="local",
        choices=["local", "correlated"],
        help="Class for readout error experiment in Qiskit.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./result",
        help="Direction for saving results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=170,
        choices=[20, 21, 30, 33, 36, 42, 43, 55, 67, 170],
        help="Seed for algorithmic, transpiling, and measurement in Qiskit.",
    )
    parser.add_argument(
        "--shots", type=int, default=1000, help="Shots for circuit execution."
    )
    print(get_context(parser.parse_args()))
