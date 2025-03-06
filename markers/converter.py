import math


# Assumes that there is no rotation
class Converter:
    def __init__(
        self,
        point_0_in_a: tuple[float, float],
        point_1_in_a: tuple[float, float],
        point_0_in_b: tuple[float, float],
        point_1_in_b: tuple[float, float],
        distance_cm: float,
    ):
        self.distance_cm = distance_cm

        # coordinate system a
        self.x0_a, self.y0_a = point_0_in_a
        self.x1_a, self.y1_a = point_1_in_a
        self.distance_px_in_a = math.fabs(self.x1_a - self.x0_a)
        self.px_to_cm_ratio_in_a = self.distance_cm / self.distance_px_in_a
        self.cm_to_px_ratio_in_a = self.distance_px_in_a / self.distance_cm

        # coordinate system b
        self.x0_b, self.y0_b = point_0_in_b
        self.x1_b, self.y1_b = point_1_in_b
        self.distance_px_in_b = math.fabs(self.x1_b - self.x0_b)
        self.px_to_cm_ratio_in_b = self.distance_cm / self.distance_px_in_b
        self.cm_to_px_ratio_in_b = self.distance_px_in_b / self.distance_cm

    def convert_from_a_to_b(self, point: tuple[float, float]):
        x, y = point
        new_x = x * self.px_to_cm_ratio_in_a * self.cm_to_px_ratio_in_b
        new_y = y * self.px_to_cm_ratio_in_a * self.cm_to_px_ratio_in_b
        return new_x, new_y

    def convert_from_b_to_a(self, point: tuple[float, float]):
        x, y = point
        new_x = x * self.px_to_cm_ratio_in_b * self.cm_to_px_ratio_in_a
        new_y = y * self.px_to_cm_ratio_in_b * self.cm_to_px_ratio_in_a
        return new_x, new_y
