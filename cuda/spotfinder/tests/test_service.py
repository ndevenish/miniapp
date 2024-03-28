import unittest
import os
from unittest.mock import Mock
from service import GPUPerImageAnalysis

class TestGPUPerImageAnalysis(unittest.TestCase):
    def test_gpu_per_image_analysis(self):
        # Creating an instance of the service
        service = GPUPerImageAnalysis()

        # Mocking the RecipeWrapper and its properties
        mock_rw = Mock()
        mock_rw.recipe_step = {"parameters": {"filename": "/dls/mx-scratch/gw56/i04-1-ins-huge/Insulin_6/Insulin_6_1.nxs"}}
        mock_rw.transport = Mock()

        # Mocking the header and message
        mock_header = Mock()
        mock_message = Mock()

        # Calling the method with the mocked parameters
        service.gpu_per_image_analysis(mock_rw, mock_header, mock_message)

        # Asserting that a JSON output file was created
        self.assertTrue(os.path.exists("output.json"))

if __name__ == '__main__':
    unittest.main()