"""Leakage and capacity invariants using small disposable fixtures."""

import unittest

import numpy as np

from swarm.clustering import balanced_assignment, route_weights
from swarm.data import document_key, evaluation_split, prefix_texts


class FakeDecoder:
    def decode_batch(self, batches):
        return [" ".join(map(str, ids)) for ids in batches]


class SwarmDataTests(unittest.TestCase):
    def test_normalized_duplicate_identity_and_fixed_partition(self):
        first = document_key("Hello   world\nagain")
        second = document_key("Hello world again")
        self.assertEqual(first, second)
        self.assertEqual(evaluation_split(first), evaluation_split(second))

    def test_future_suffix_cannot_enter_router_text(self):
        first = np.arange(514).reshape(2, 257)
        second = first.copy()
        second[:, 128:] += 5000
        self.assertEqual(prefix_texts(FakeDecoder(), first), prefix_texts(FakeDecoder(), second))

    def test_capacity_balancing_covers_every_block_once(self):
        vectors = np.array([[0., 0.], [0., .1], [0., .2], [0., .3], [10., 0.], [11., 0.]], dtype=np.float32)
        groups = balanced_assignment(vectors, np.array([[0., 0.], [10., 0.]]), np.arange(6))
        np.testing.assert_array_equal(np.bincount(groups), [3, 3])
        self.assertEqual(len(groups), len(vectors))
        with self.assertRaises(ValueError):
            balanced_assignment(vectors[:5], np.array([[0., 0.], [10., 0.]]), np.arange(5))

    def test_sparse_router_normalization_and_deterministic_tie(self):
        vectors = np.array([[0., 0.], [2., 0.]])
        centers = np.array([[0., 0.], [0., 0.], [2., 0.]])
        weights = route_weights(vectors, centers, top_k=1)
        np.testing.assert_array_equal(weights, [[1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(route_weights(vectors, centers).sum(1), [1., 1.])


if __name__ == "__main__":
    unittest.main()
