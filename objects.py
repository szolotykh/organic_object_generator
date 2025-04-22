import numpy as np
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def generate_points_inside(self, num_points):
        pass

    @abstractmethod
    def generate_points_on_surface(self, num_points):
        pass


class Sphere(Shape):
    def __init__(self, radius):
        self.radius = radius

    def generate_points_inside(self, num_points):
        points = []
        while len(points) < num_points:
            p = np.random.uniform(-self.radius, self.radius, size=3)
            if p.dot(p) <= self.radius**2:
                points.append(p)
        return np.array(points)

    def generate_points_on_surface(self, num_points):
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

    def generate_points_inside(self, num_points):
        points = []
        while len(points) < num_points:
            x, y = np.random.uniform(-self.radius, self.radius, size=2)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            if x**2 + y**2 <= self.radius**2:
                points.append([x, y, z])
        return np.array(points)

    def generate_points_on_surface(self, num_points):
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

    def generate_points_inside(self, num_points):
        points = []
        while len(points) < num_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(self.inner_radius**2, self.outer_radius**2))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            points.append([x, y, z])
        return np.array(points)

    def generate_points_on_surface(self, num_points):
        points = []
        while len(points) < num_points:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.choice([self.inner_radius, self.outer_radius])
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = np.random.uniform(-self.height / 2, self.height / 2)
            points.append([x, y, z])
        return np.array(points)
