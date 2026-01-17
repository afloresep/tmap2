def test_tmap_exports_lshforest():
    from tmap import LSHForest

    assert LSHForest.__name__ == "LSHForest"

def test_tmap_exports_minhash():
    from tmap import MinHash 

    assert MinHash.__name__ == "MinHash"

def test_tmap_exports_weighted_minhash():
    from tmap import WeightedMinHash 

    assert WeightedMinHash.__name__ == "WeightedMinHash"
