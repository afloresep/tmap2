from numpy.typing import NDArray
from dataclasses import dataclass, field
import numpy as np
from typing import TypeAlias


#slots=True so no more instances can be generated, aslo probably more efficient than with __dict__
@dataclass(slots=True)
class _TrieNode:
    """Internal trie node for LSH Forest"""
    children: dict[int, "_TrieNode"] = field(default_factory=dict)  # mutable
    indices: list[int] = field(default_factory=list)


@dataclass
class _LSHForestState:
    """Internal state of the LSH forest """
    trees: list[_TrieNode] # l prefix trees
    permutations: NDArray[np.int32] # (l,d) permutation indices
    signatures: NDArray[np.uint64] | None # (n, d) if store =True
    n_indexed: int # NUmber of indexed signatures
    is_indexed: bool =False # whether index() has been called or not



class LSHForest:
    """
    LSH Forest data structure for approximate nearest neighbor search.

    Incorporates optional linear scan to increase recovery performance.
    Most query methods are available in parallelized versions with `batch_` prefix.

    Args:
        d: Dimensionality of MinHash vectors (number of permutations). Default: 128
        l: Number of prefix trees. Default: 8
        store: Store signatures for enhanced querying (get_hash, distance_by_id). Default: True
        weighted: Whether using weighted MinHash signatures. Default: False

    Example:
        >>> from tmap import MinHash, LSHForest
        >>>
        >>> # Create MinHash signatures
        >>> mh = MinHash(num_perm=128)
        >>> sigs = mh.encode(fingerprints)
        >>>
        >>> # Build LSH Forest
        >>> lsh = LSHForest(d=128, l=8)
        >>> lsh.batch_add(sigs)
        >>> lsh.index()
        >>>
        >>> # Query
        >>> neighbors = lsh.query(sigs[0], k=10)
        >>>
        >>> # Or use linear scan for better accuracy
        >>> results = lsh.query_linear_scan(sigs[0], k=10, kc=10)

    """
    def __init__(self,
                d: int = 128,
                l: int= 8,
                store:bool = True,
                weighted: bool = False) -> None:
        self._LSHForestState =  _LSHForestState
        self._LSHForestState.permutations = np.array([l,d])
        self.d = d
        self.l = l
        self.weighted = weighted
        


    def add(self, signature: NDArray[np.uint64]) -> None:
        """
        Add a MinHash signature to the LSH forest.

        Args:
             signature: MinHash vector of shape (d,) or (d, 2) for weighted

        Note:
             Call index() after adding signatures to build/update the index.
        """
        assert ((signature.ndim == 1 and signature.shape[0] > 0) \
               or (signature.ndim == 2 and signature.shape[0] > 0 and signature.shape[1] == 2)),\
                  "Signature must be of shape (d,) or (d,2) for weighted. For multiple signatures use `batch_add`"

        #set indexed to True
        self._LSHForestState.is_indexed = True 

    def batch_add(self, signatures: NDArray[np.uint64]) -> None:
        """Add multiple MinHash signatures to the LSH Forest (optimized)

        Args:
            signatures (NDArray[np.uint64]): MinHash vectors of shape (n,d) or (n,d,2) for weighted

        Note: 
            call index() after adding signatures to build/update the index
        """
        assert ((len(signatures.shape) == 2) or (len(signatures.shape) ==3 and signatures.shape[-1]==2)),\
              f"The shape of signatures must be (n,d) or (n,d,2) but got {signatures.shape} instead" 
        #set indexed to True
        self._LSHForestState.is_indexed = True 


    def index(self) -> None:
        """
        Build/rebuild the LSH forest index.

        Must be called after adding signatures with add() or batch_add().
        Can be called multiple times as new data is added.
        """
        assert (self._LSHForestState.is_indexed), "LSHForest must be called after adding signatures with add() or batch_add()"


     
