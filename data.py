import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class RALO_Dataset(Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 subset="train",
                 transform=None,
                 score="GE",
    ):
        super(RALO_Dataset, self).__init__()

        self.transform = transform
        self.subset = subset
	      self.score = score
        csv = pd.read_csv(csvpath, skiprows=1)
        if score == "GE":
          csv["Geographic Extent"] = (csv["Total GEOGRAPHIC"] + csv["Total GEOGRAPHIC.1"])/2
          self.labels = csv["Geographic Extent"]
	        self.labels = np.asarray(self.labels)
	        self.labels = np.asarray(self.labels, dtype = np.float32)
	      else:
	        csv["Lung Opacity"] = (csv["Total OPACITY"] + csv["Total OPACITY.1"])/2
	        self.labels = np.asarray(self.labels)
	        self.olabels = csv["Lung Opacity"]
          self.labels = np.asarray(self.labels, dtype = np.float32)
        
	      self.images = [os.path.join(imgpath, str(idx) + ".jpg") for idx in range(0, len(self.labels))]
       
        test_ids = [0,   14,   15,   16,   22,   23,   25,   26,   27,   28,   29,
                    30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,
                    41,   42,   43,   61,   62,   63,   64,   65,   66,   74,   75,
                    76,   77,   78,   79,   80,   81,   82,   84,  111,  112,  113,
                    162,  163,  164,  165,  166,  167,  168,  169,  173,  174,  180,
                    181,  257,  259,  260,  261,  262,  263,  264,  265,  266,  267,
                    268,  269,  270,  271,  272,  273,  274,  275,  276,  277,  312,
                    314,  315,  316,  317,  318,  319,  320,  321,  322,  323,  324,
                    325,  326,  327,  328,  329,  330,  331,  404,  405,  406,  407,
                    417,  418,  419,  420,  421,  422,  423,  424,  425,  426,  427,
                    428,  429,  430,  431,  432,  433,  434,  435,  471,  474,  475,
                    476,  477,  478,  517,  530,  531,  532,  533,  534,  564,  565,
                    566,  572,  606,  613,  659,  660,  661,  662,  663,  664,  665,
                    668,  771,  772,  773,  774,  775,  776,  777,  781,  782,  783,
                    784,  786,  787,  788,  789,  790,  791,  792,  793,  794,  795,
                    796,  797,  798,  799,  800,  801,  802,  810,  869,  883,  884,
                    885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  896,
                    897,  898,  899,  916,  917,  918,  919,  920,  921,  922,  923,
                    924,  925,  926,  927,  930,  932,  976,  977,  978,  979,  983,
                    984,  985,  986,  987,  988,  989,  990,  991,  992, 1010, 1011,
                    1071, 1083, 1084, 1085, 1086, 1087, 1107, 1108, 1126, 1127, 1128,
                    1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1151, 1152,
                    1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163,
                    1164, 1165, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175,
                    1176, 1177, 1179, 1221, 1222, 1223, 1224, 1225, 1226, 1238, 1239,
                    1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281,
                    1282, 1283, 1284, 1285, 1286, 1287, 1316, 1324, 1325, 1326, 1327,
                    1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338,
                    1339, 1340, 1341, 1342, 1373, 1374, 1375, 1376, 1377, 1378, 1379,
                    1380, 1381, 1382, 1383, 1384, 1385, 1386, 1507, 1508, 1509, 1527,
                    1529, 1530, 1531, 1532, 1561, 1562, 1587, 1598, 1599, 1600, 1601,
                    1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1627, 1671, 1672,
                    1673, 1674, 1675, 1676, 1677, 1682, 1683, 1692, 1693, 1694, 1695,
                    1696, 1697, 1698, 1699, 1715, 1745, 1749, 1762, 1763, 1772, 1800,
                    1801, 1802, 1803, 1813, 1814, 1817, 1818, 1819, 1820, 1821, 1822,
                    1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833,
                    1834, 1869, 1870, 1871, 1872, 1876, 1877, 1878, 1879, 1880, 1881,
                    1910, 1936, 1937, 1938, 1941, 1942, 1943, 1944, 1945, 1946, 1947,
                    1948, 1949, 1950, 1961, 1978, 1979, 1980, 2009, 2017, 2018, 2019,
                    2020, 2021, 2022, 2069, 2070, 2071, 2072, 2073, 2074, 2108, 2109,
                    2155, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2182, 2183,
                    2189, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204,
                    2213, 2214, 2215, 2216, 2217, 2218, 2225, 2255, 2257, 2258, 2259,
                    2260, 2261, 2262, 2263, 2264, 2265, 2277, 2278, 2279, 2285, 2286,
                    2287, 2288, 2289, 2291, 2340, 2342, 2345, 2354, 2355, 2356, 2358]

        if subset == "train":
            self.labels = [self.labels[idx] for idx in range(0, len(self.labels)) if idx not in test_ids]
            self.images = [self.images[idx] for idx in range(0, len(self.images)) if idx not in test_ids]
        else:
            self.labels = [self.labels[idx] for idx in test_ids]
            self.images = [self.images[idx] for idx in test_ids]

    def __str__(self):
        return "RALO_Dataset({}): {} images".format(self.subset, len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    dataset = RALO_Dataset(imgpath="CXR_images_scored/", csvpath="ralo-dataset-metadata.csv")
    print(dataset[0])
    print(len(dataset))

