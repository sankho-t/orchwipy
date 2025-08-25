import argparse
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger("root")
log.setLevel(logging.INFO)

from orchwipy.cmd import command_line_args, command_line_invoke
from orchwipy import Functions, MicroFunctions, ConditionalReturns, ReturnUpdates

orch = MicroFunctions(
    beginner="welcome"
)  # or Functions if pipeline will always run locally


@orch.fn()
def welcome(*, greeting: str, choice: str | None = None, **kwargs):
    log.info(f"{greeting}")
    msg = "Would you like to have tea(t) or coffee(c)?"
    if not choice:
        choice = input(msg)
    else:
        log.info(msg + choice)
    return ConditionalReturns(
        choice,
        ReturnUpdates("nochoice"),
        t=ReturnUpdates("making_tea", "*"),
        c=ReturnUpdates("making_coffee", "*"),
    )


@orch.fn()
def making_tea(**kwargs):
    log.info("Let's make some tea")
    return ReturnUpdates("boil_water", choice="tea")


@orch.fn()
def making_coffee(**kwargs):
    log.info("Let's make some coffee")
    return ReturnUpdates("boil_water", choice="coffee", pour_slow=True)


@orch.fn(terminates=True)
def nochoice(**kwargs):
    log.info("Bye")
    return {}


@orch.fn()
def boil_water(*, choice: str, **kwargs):
    log.info("Boil some water")
    return ConditionalReturns(
        choice, ReturnUpdates("dip"), coffee=ReturnUpdates("grind")
    )


@orch.fn()
def grind(**kwargs):
    log.info("Grind some coffee beans")
    return ReturnUpdates("pour_water", target="coffee grounds on a filter paper")


@orch.fn(default_next="pour_water")
def dip(**kwargs):
    log.info("Dip a tea bag")
    return ReturnUpdates("pour_water", target="cup")


@orch.fn(default_next="drink")
def pour_water(*, pour_slow: bool = False, target: str, **kwargs):
    log.info(f"Pour the water{' slowly' if pour_slow else ''} onto the {target}")
    return ReturnUpdates("drink")


@orch.fn(terminates=True)
def drink(**kwargs):
    log.info("Now enjoy")
    return {}


def lambda_handler(event, context):
    # redirect to orchwipy's lambda handler
    return orch.lambda_handler(event, context)


argp = argparse.ArgumentParser("Example pipelining with orchwipy")

command_line_args(argp=argp)

if __name__ == "__main__":
    args = argp.parse_args()
    command_line_invoke(orch, args)
