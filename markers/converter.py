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
        self.cm_distance = distance_cm

        # coordinate system a
        self.x0_a, self.y0_a = point_0_in_a
        self.x1_a, self.y1_a = point_1_in_a
        self.pixel_distance_in_a = math.fabs(self.x1_a - self.x0_a)
        self.cm_per_pixel_in_a = self.cm_distance / self.pixel_distance_in_a
        self.pixel_per_cm_in_a = self.pixel_distance_in_a / self.cm_distance

        # coordinate system b
        self.x0_b, self.y0_b = point_0_in_b
        self.x1_b, self.y1_b = point_1_in_b
        self.pixel_distance_in_b = math.fabs(self.x1_b - self.x0_b)
        self.cm_per_pixel_in_b = self.cm_distance / self.pixel_distance_in_b
        self.pixel_per_cm_in_b = self.pixel_distance_in_b / self.cm_distance

    def convert_from_a_to_b(self, point: tuple[float, float]):
        x, y = point
        new_x = x * self.cm_per_pixel_in_a * self.pixel_per_cm_in_b
        new_y = y * self.cm_per_pixel_in_a * self.pixel_per_cm_in_b
        return new_x, new_y

    def convert_from_b_to_a(self, point: tuple[float, float]):
        x, y = point
        new_x = x * self.cm_per_pixel_in_b * self.pixel_per_cm_in_a
        new_y = y * self.cm_per_pixel_in_b * self.pixel_per_cm_in_a
        return new_x, new_y


# cm per pixel is not quite accurate because this cm distance is between the marker points and there are further pixels at the edges

# in input coordinate system
# use marker location marker0 in top left
# x = x-coord - x-marker0
# y = y-coord - y-marker0
# x, y is now the the coordinate of the point in the original coordinate system but using marker0 as origin

# in output coordinate system
# use marker location marker0 in top left

# maybe rotation matrix?
# for input coordinate system
# take coordinates of two markers in horizontal plane
# determine angle based on cm-distance and y-pixel shift
# do the same for output coordinate system
# now subtract the 2nd angle from the first and that's your theta
# rotation_matrix = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
# [[x * cos(theta) - y * sin(theta)], [x * sin(theta) + y * cos(theta)]]
# will give you the vector [[rotated_x], [rotated_y]]
