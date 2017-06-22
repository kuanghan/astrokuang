import unittest
import numpy as np
import photom_utils as pu


class PhotomUtilTestCase(unittest.TestCase):
    """Tests for photom_utils.py"""

    def assertSequenceAlmostEqual(self, a, b, places=7, msg=None):
        """
        My own implementation to test if every corresponding element in a
        and b are approximately equal.
        """
        msg = self._formatMessage(msg,
            "{} and {} are not approximately equal".format(a, b))
        if len(a) != len(b):
            raise self.failureException(msg)
        for i in range(len(a)):
            if np.round(a[i], places) != np.round(b[i], places):
                raise self.failureException(msg)

    def test_flux2mag_detection_number(self):
        """Test if flux2mag returns the right number for detections"""
        self.assertAlmostEqual(pu.flux2mag(2.0, 23.9), 23.9-2.5*np.log10(2.))

    def test_flux2mag_nondetection_number(self):
        """Test if flux2mag returns the right number for non-detections"""
        self.assertAlmostEqual(pu.flux2mag(-1.0, 23.9), 99.0)

    def test_flux2mag_nondetection_array(self):
        """
        Test if flux2mag returns the correct array for both detection &
        non-detections.
        """
        self.assertSequenceAlmostEqual(pu.flux2mag([-1.0, 1.0], 23.9),
                                       [99.0, 23.9])

    def test_ABmag2uJy_detection(self):
        """
        Test if ABmag2uJy returns the correct flux in uJy when
        input is a detected magnitude (< 99 mag).
        """
        self.assertAlmostEqual(pu.ABmag2uJy(20), 36.3078054770101)

    def test_ABmag2uJy_nondetection(self):
        """
        Test if ABmag2uJy properly returns zero when magnitude is more than 99
        """
        self.assertAlmostEqual(pu.ABmag2uJy(99), 0.)

    def test_ABmag2uJy_arrayType(self):
        """
        Test if ABmag2uJy properly returns a numpy array when a list of
        magnitudes are input
        """
        self.assertIsInstance(pu.ABmag2uJy([20, 20]), np.ndarray)

    def test_ABmag2uJy_arrayValues(self):
        """
        Test if ABmag2uJy properly returns the correct list of flux densities
        """
        self.assertSequenceAlmostEqual(pu.ABmag2uJy([20, 23.9]),
                                       [36.3078054770101, 1.0])

    def test_ABmag2uJy_eazy_detection(self):
        """Test if ABmag2uJy_eazy returns the correct string"""
        ans = "1.000000e+00  9.647820e-02"
        self.assertEqual(pu.ABmag2uJy_eazy(23.9, 0.1), ans)


if __name__ == "__main__":
    unittest.main(verbosity=1)