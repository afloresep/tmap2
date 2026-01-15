# Design Patterns Guide for TreeMap

This document explains the design patterns used in TreeMap and when to use them.

## 1. Decorators

Decorators are functions that modify other functions/methods. Python has several built-in decorators you'll use frequently.

### @property - Computed Attributes

Makes a method behave like an attribute (no parentheses needed):

```python
class Index:
    def __init__(self):
        self._is_built = False

    @property
    def is_built(self) -> bool:
        """Access with: index.is_built (not index.is_built())"""
        return self._is_built
```

**When to use:**
- Computed values that feel like attributes
- Read-only access to internal state
- Lazy computation (compute on first access)

### @abstractmethod - Interface Contracts

Forces subclasses to implement a method:

```python
from abc import ABC, abstractmethod

class Index(ABC):
    @abstractmethod
    def build(self, data) -> None:
        """Subclasses MUST implement this."""
        ...  # No implementation here

class FaissIndex(Index):
    def build(self, data) -> None:
        # Must implement or TypeError on instantiation
        pass
```

**When to use:**
- Defining interfaces/contracts
- When there's no sensible default implementation
- Core methods that vary per implementation

### @classmethod - Alternative Constructors

Receives the class (not instance) as first argument:

```python
class Index:
    def __init__(self, seed: int):
        self.seed = seed

    @classmethod
    def from_file(cls, path: str) -> "Index":
        """Factory method - creates instance from file."""
        data = load_file(path)
        return cls(seed=data["seed"])  # cls is Index (or subclass)

# Usage:
index = FaissIndex.from_file("index.pkl")  # Returns FaissIndex
```

**When to use:**
- Loading from files
- Creating instances with different input formats
- Factory patterns

### @staticmethod - Pure Functions in Class Namespace

Receives nothing (no self, no cls):

```python
class MSTBuilder:
    @staticmethod
    def kruskal(edges: list) -> list:
        """Pure function - doesn't use any instance state."""
        return sorted(edges, key=lambda e: e.weight)
```

**When to use:**
- Utility functions related to the class
- Pure functions that don't need instance state
- Often better as module-level functions instead

### Custom Decorators

You can create your own decorators for cross-cutting concerns:

```python
import functools
import time

def timing(func):
    """Log how long a function takes."""
    @functools.wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

class Layout:
    @timing
    def compute(self, tree):
        # This will print: "compute took 1.23s"
        ...
```

**Common custom decorators:**
- `@timing` - Performance logging
- `@retry(n=3)` - Retry on failure
- `@cache` - Memoization (use `functools.cache` in Python 3.9+)
- `@validate_input` - Input validation

---

## 2. Abstract Base Classes (ABC)

ABCs define interfaces - WHAT something does, not HOW.

```python
from abc import ABC, abstractmethod

class Index(ABC):
    """Contract: All indices must have these methods."""

    @abstractmethod
    def build(self, data) -> None: ...

    @abstractmethod
    def query(self, point, k: int) -> list: ...

# This will raise TypeError:
class BadIndex(Index):
    pass  # Forgot to implement build() and query()

bad = BadIndex()  # TypeError: Can't instantiate abstract class
```

**ABC vs Protocol:**
- ABC: Explicit inheritance required (`class Foo(Index)`)
- Protocol: Any class with matching methods works (duck typing)

Use ABC when:
- You have shared implementation (methods all subclasses use)
- You want isinstance() checks to work
- Explicit "is-a" relationship is important

---

## 3. Strategy Pattern

Different algorithms with the same interface:

```python
# Define the strategy interface
class Layout(ABC):
    @abstractmethod
    def compute(self, tree) -> Coordinates: ...

# Concrete strategies
class ForceDirectedLayout(Layout):
    def compute(self, tree) -> Coordinates:
        # Force simulation algorithm
        ...

class RadialLayout(Layout):
    def compute(self, tree) -> Coordinates:
        # Concentric circles algorithm
        ...

# Usage - swap strategies without changing client code
def visualize(tree, layout: Layout):
    coords = layout.compute(tree)  # Works with any Layout
    return render(coords)

visualize(tree, ForceDirectedLayout())  # or RadialLayout()
```

**When to use:**
- Multiple algorithms for the same task
- You want to swap implementations at runtime
- Algorithms should be independently testable

---

## 4. Composition over Inheritance

**Composition (HAS-A):**
```python
class TreeMap:
    def __init__(self, index: Index, layout: Layout):
        self.index = index    # TreeMap HAS-A Index
        self.layout = layout  # TreeMap HAS-A Layout
```

**Inheritance (IS-A) - often worse:**
```python
class TreeMap(Index, Layout):  # TreeMap IS-A Index AND Layout?
    # Multiple inheritance is confusing
    # Can't easily swap Index or Layout implementations
```

**Why composition is better:**
- Flexibility: Mix any Index with any Layout
- Clarity: Clear ownership/responsibility
- Testability: Mock individual components
- No "diamond problem" from multiple inheritance

---

## 5. Dependency Injection

Pass dependencies in, don't create them internally:

```python
# Good: Dependencies injected
class TreeMap:
    def __init__(self, index: Index, layout: Layout):
        self.index = index
        self.layout = layout

tm = TreeMap(
    index=FaissIndex(seed=42),  # User chooses
    layout=ForceDirectedLayout(),  # User chooses
)

# Bad: Dependencies hardcoded
class TreeMap:
    def __init__(self):
        self.index = FaissIndex()  # Can't change this!
        self.layout = ForceDirectedLayout()  # Can't mock in tests!
```

**Benefits:**
- Configurable: Users choose implementations
- Testable: Easy to inject mocks
- Flexible: Change behavior without changing code

---

## 6. Fluent API (Method Chaining)

Return `self` to enable chaining:

```python
class TreeMap:
    def build_from_vectors(self, vectors) -> "TreeMap":
        self.index.build(vectors)
        return self  # Enable chaining

    def compute(self, k: int) -> "TreeMap":
        self._knn = self.index.query_knn(k)
        return self

    def get_coordinates(self) -> Coordinates:
        return self._coordinates

# Usage - reads like English
coords = (
    TreeMap(index, layout)
    .build_from_vectors(data)
    .compute(k=20)
    .get_coordinates()
)
```

---

## 7. Type Hints

Use type hints everywhere for:
- IDE autocomplete
- Catch bugs before runtime
- Self-documenting code

```python
from typing import Self  # Python 3.11+

class Index:
    def build(self, data: NDArray[np.float32]) -> Self:
        ...

    def query(self, k: int) -> KNNGraph:
        ...

# Union types (3.10+)
def process(data: list[int] | NDArray) -> None:
    ...

# Optional (can be None)
def find(label: str) -> Node | None:
    ...
```

---

## Quick Reference: When to Use What

| Pattern | Use When |
|---------|----------|
| @property | Computed attributes, read-only access |
| @abstractmethod | Defining required interface methods |
| @classmethod | Factory methods, alternative constructors |
| ABC | Shared base implementation + required interface |
| Protocol | Duck typing with type checking |
| Strategy | Multiple swappable algorithms |
| Composition | Combining capabilities from multiple classes |
| DI | Components should be configurable/testable |
| Fluent API | Sequential operations that read naturally |
