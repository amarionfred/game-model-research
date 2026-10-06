"""Budget, parent and information-boundary checks; no trained-model claim."""

import unittest

import numpy as np

from swarm.model import BASE_MODELS, ModelConfig
from swarm.run_grid import BATCH, TOKENS, schedule, training_order


class GridTests(unittest.TestCase):
    def test_standalone_system_cost_and_dense_rounding(self):
        for family, base in BASE_MODELS.items():
            tasks = schedule(family)
            self.assertEqual(len(tasks), 99)
            self.assertEqual(len({t["id"] for t in tasks}), 99)
            seen = set()
            for task in tasks:
                if task["parent"]:
                    self.assertIn(task["parent"], seen)
                seen.add(task["id"])
            for seed in (17, 29, 43):
                selected = [t for t in tasks if t["seed"] == seed]
                warm = next(t for t in selected if t["arm"] == "warm")
                for method in ("clustered", "random"):
                    for k in (2, 4, 8):
                        branches = [t for t in selected if t["arm"] == method and t["k"] == k]
                        self.assertEqual(len(branches), k)
                        tokens = (warm["steps"] + sum(t["steps"] for t in branches)) * BATCH * 256
                        self.assertEqual(tokens, TOKENS)
                target = TOKENS * base.train_matmul_flops_per_token
                for t in selected:
                    if t["arm"] == "calibrator":
                        config = ModelConfig(**t["config"])
                        batch_flops = BATCH * 256 * config.train_matmul_flops_per_token
                        self.assertLessEqual(t["steps"] * batch_flops, target)
                        self.assertLess(target - t["steps"] * batch_flops, batch_flops)

    def test_branch_orders_partition_only_continuation_data(self):
        pools = {"warm": np.arange(131072), "continuation": np.arange(131072, 262144)}
        groups = {f"{method}-k{k}": {"block_ids": pools["continuation"],
                   "assignments": np.arange(131072) % k}
                  for method in ("clustered", "random") for k in (2, 4, 8)}
        tasks = [t for t in schedule("small") if t["seed"] == 17]
        for k in (2, 4, 8):
            selected = [t for t in tasks if t["arm"] == "clustered" and t["k"] == k]
            orders = [training_order(t, pools, groups)[0] for t in selected]
            np.testing.assert_array_equal(np.sort(np.concatenate(orders)), pools["continuation"])
            for task, order in zip(selected, orders):
                np.testing.assert_array_equal(order, training_order(task, pools, groups)[0])
                self.assertEqual(len(order), task["steps"] * BATCH)
        for task in tasks:
            order, _ = training_order(task, pools, groups)
            self.assertEqual(len(order), len(np.unique(order)))
            self.assertTrue(np.all((order >= 0) & (order < 262144)))


if __name__ == "__main__":
    unittest.main()
