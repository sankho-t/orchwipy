from copy import deepcopy
import unittest

from orchwipy.call_graph import MissingArgument
from sample import orch, ReturnUpdates

orch_fail = deepcopy(orch)


@orch_fail.fn()
def dip(*, foo: str, **kwargs):
    print("Dip a tea bag")
    return ReturnUpdates("pour_water", target="cup")


class TestFailOnArgumentMissing(unittest.TestCase):
    def test_0(self):
        with self.assertWarns(MissingArgument):
            orch_fail.check_graph("welcome")
