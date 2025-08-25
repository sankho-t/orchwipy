import argparse
import json
import importlib
import tempfile
from pathlib import Path

from .func import Functions
from .microservice import MicroFunctions, build_cfn_template


def command_line_args(argp: argparse.ArgumentParser):
    args = argp.add_subparsers(title="orchwipy features")
    runarg = args.add_parser("run")
    runarg.add_argument("--run", action="store_true")
    runarg.add_argument(
        "--arg-file",
        type=argparse.FileType("r"),
        help="JSON file to load the arguments for the first function, requires --run",
    )
    validarg = args.add_parser("check")
    validarg.add_argument("--check-call-graph", action="store_true")
    validarg.add_argument("--save-call-graph", type=argparse.FileType("w"))

    lambdaarg = args.add_parser(
        "lambda",
        description="Create lambda template. Use the MicroFunctions.build_cfn_template for options",
    )
    lambdaarg.add_argument(
        "--create-template",
        type=argparse.FileType("w"),
        help="Where to place the generated template",
        required=True,
    )
    lambdaarg.add_argument(
        "--code-path",
        type=argparse.FileType("r"),
        help="Relative (from template) path to python file where the lambda_handler redirect is located",
        required=True,
    )


def command_line_invoke(pipe: Functions | MicroFunctions, args: argparse.Namespace):
    args_ = vars(args)

    if save_to := args_.get("save_call_graph"):
        pipe.check_graph(pipe.beginner, warn_default_passing=True)
        pipe.save_dot(save_to)
        save_to.close()

    elif args_.get("check_call_graph"):
        pipe.check_graph(pipe.beginner, warn_default_passing=True)
        with tempfile.NamedTemporaryFile("wt") as f:
            pipe.save_dot(f, top_node=pipe.beginner)

    if args_.get("run"):
        if argf := args.arg_file:
            args__ = json.load(argf)
            pipe.run(**args__)
            argf.close()
        else:
            pipe.run()

    elif args_.get("arg_file"):
        raise ValueError("--run required with --arg-file")

    if args_.get("create_template"):
        assert isinstance(pipe, MicroFunctions)

        pipe.check_graph(pipe.beginner)

        build_cfn_template(
            output_template=args.create_template,
            code_py_relative_to_template=(
                Path(args.code_path.name).relative_to(
                    Path(args.create_template.name).parent
                )
            ),
            plant=pipe,
            environment=None,
            layers=[],
            more_params={},
            more_template_params=None,
            policies=[],
        )

        args.create_template.close()
        args.code_path.close()
