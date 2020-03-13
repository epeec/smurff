import unittest
import smurff

class TestHDF5(unittest.TestCase):
    def test_simple(self):
        predict_session = smurff.PredictSession("train_output.h5")

        # predict all
        p1 = predict_session.predict_all()
        self.assertEqual(p1.shape, (predict_session.num_samples,)  + predict_session.data_shape)

        # predict some
        Ytest = predict_session.h5_file["config/test/data"][()]
        p2 = sorted(predict_session.predict_some(Ytest))
        self.assertEqual(len(p2), Ytest.nnz)

        p3 = predict_session.predict_one(p2[0].coords, p2[0].val)
        self.assertEqual(p3.coords, p2[0].coords)
        self.assertAlmostEqual(p3.val, p2[0].val, places = 2)
        self.assertAlmostEqual(p3.pred_1sample, p2[0].pred_1sample, places = 2)
        self.assertAlmostEqual(p3.pred_avg, p2[0].pred_avg, places = 2)

        # check predict_session.predict_some vs predict_session.predict_all
        for s in p2:
            ecoords = (Ellipsis,) + s.coords
            for p in zip(s.pred_all, p1[ecoords]):
                self.assertAlmostEqual(*p, places=2)

        ytest_rmse_avg = predict_session.statsYTest()["rmse_avg"]
        p2_rmse_avg = smurff.calc_rmse(p2)

        self.assertAlmostEqual(ytest_rmse_avg, p2_rmse_avg, places = 2)

if __name__ == '__main__':
    unittest.main()