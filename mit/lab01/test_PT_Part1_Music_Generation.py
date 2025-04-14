import torch
import unittest
from .PT_Part1_Music_Generation import DenseLayer
from .PT_Part1_Music_Generation import compute

class TestDenseLayer(unittest.TestCase):
    def test_forward_pass(self):
        num_inputs = 2
        num_outputs = 3
        layer = DenseLayer(num_inputs, num_outputs)
        x_input = torch.tensor([1.0, 2.0])
        y_output = layer(x_input)

        # Check output shape
        self.assertEqual(y_output.shape, torch.Size([num_outputs]))

        # Check if output values are between 0 and 1 (sigmoid range)
        self.assertTrue(torch.all(y_output >= 0) and torch.all(y_output <= 1))

class TestComputeFunction(unittest.TestCase):
    def test_compute(self):
        a = 1.5
        b = 2.5
        result = compute(a, b)

        # Expected result: (a + b) * (b - 1)
        expected_result = (a + b) * (b - 1)
        self.assertAlmostEqual(result, expected_result, places=5)

if __name__ == "__main__":
    unittest.main()