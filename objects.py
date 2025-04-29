import numpy as np
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def generate_points_inside(self, num_points, seed=None):
        pass

    @abstractmethod
    def generate_points_on_surface(self, num_points, seed=None):
        pass

    def find_nearest_pairs(self, points, k_nn):
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, idxs = tree.query(points, k=k_nn+1)
        pairs = set()
        for i in range(len(points)):
            for j in idxs[i, 1:]:
                pairs.add(tuple(sorted((i, j))))
        return list(pairs)

    def find_random_pairs(self, num_points, k_nn, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        pairs = set()
        while len(pairs) < num_points * k_nn:
            i, j = np.random.choice(num_points, size=2, replace=False)
            pairs.add(tuple(sorted((i, j))))
        return list(pairs)

    def find_connection_pairs(self, points, connection_type, k_nn, seed=None):
        if connection_type == "nearest":
            pairs = self.find_nearest_pairs(points, k_nn)
        elif connection_type == "random":
            pairs = self.find_random_pairs(len(points), k_nn, seed=seed)  # Pass seed
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")

        # Validate all pairs using _is_valid_connection
        valid_pairs = [
            pair for pair in pairs
            if self._is_valid_connection(points[pair[0]], points[pair[1]])
        ]
        return valid_pairs

    def _is_valid_connection(self, p1, p2):
        """Default validation: Always valid. Override in subclasses if needed."""
        return True


class Sphere(Shape):
    def __init__(self, radius):
        self.radius = radius

    def generate_points_inside(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            p = np.random.uniform(-self.radius, self.radius, size=3)
            if p.dot(p) <= self.radius**2:
                points.append(p)
        return np.array(points)

    def generate_points_on_surface(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            p = np.random.normal(size=3)
            p /= np.linalg.norm(p)  # Normalize to unit vector
            p *= self.radius        # Scale to sphere surface
            points.append(p)
        return np.array(points)


class Cylinder(Shape):
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    def generate_points_inside(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            x, y = np.random.uniform(-self.radius, self.radius, size=2)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            if x**2 + y**2 <= self.radius**2:
                points.append([x, y, z])
        return np.array(points)

    def generate_points_on_surface(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            angle = np.random.uniform(0, 2 * np.pi)
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            points.append([x, y, z])
        return np.array(points)


class Tube(Shape):
    def __init__(self, inner_radius, outer_radius, height):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.height = height

    def generate_points_inside(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(self.inner_radius**2, self.outer_radius**2))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            points.append([x, y, z])
        return np.array(points)

    def generate_points_on_surface(self, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set seed for reproducibility
        points = []
        while len(points) < num_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.choice([self.inner_radius, self.outer_radius])
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            points.append([x, y, z])
        return np.array(points)

    def _is_valid_connection(self, p1, p2):
        """Check if the connection between p1 and p2 avoids the central core."""
        # Project the points onto the XY plane
        p1_xy = np.array([p1[0], p1[1]])
        p2_xy = np.array([p2[0], p2[1]])

        # Check if both points are outside the inner radius
        if np.linalg.norm(p1_xy) >= self.inner_radius and np.linalg.norm(p2_xy) >= self.inner_radius:
            # Check if the line segment between p1 and p2 intersects the core
            t = np.linspace(0, 1, 100)  # Parameterize the line segment
            line_points = (1 - t)[:, None] * p1_xy + t[:, None] * p2_xy
            distances = np.linalg.norm(line_points, axis=1)
            if np.all(distances >= self.inner_radius):
                return True  # Valid connection
        return False  # Invalid connection
