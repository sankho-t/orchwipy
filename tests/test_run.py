import logging
import unittest

from sample import orch


class TestRun(unittest.TestCase):
    def test_output_branch_1(self):
        target = """Hi there
Would you like to have tea(t) or coffee(c)?c
Let's make some coffee
Boil some water
Grind some coffee beans
Pour the water slowly onto the coffee grounds on a filter paper
Now enjoy"""

        log = logging.getLogger("root")
        with self.assertLogs(log) as logs:
            orch.run(greeting="Hi there", choice="c")

        self.assertListEqual(
            [x.message for x in logs.records],
            target.splitlines(keepends=False),
        )

    def test_output_branch_2(self):
        target = """Hi there
Would you like to have tea(t) or coffee(c)?t
Let's make some tea
Boil some water
Dip a tea bag
Pour the water onto the cup
Now enjoy"""

        log = logging.getLogger("root")
        with self.assertLogs(log) as logs:
            orch.run(greeting="Hi there", choice="t")

        self.assertListEqual(
            [x.message for x in logs.records],
            target.splitlines(keepends=False),
        )
