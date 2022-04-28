import numpy as np
import time

# Parameters for the pure pursuit controller

POSITION_THRESHOLD_CURVE = 0.01         # Threshold value that is used to determine whether a curve/corner is coming
POSITION_THRESHOLD_STRAIGHT = 0.0006    # Threshold value that is used to determine whether a straight line is coming

# Expert's velocity and steering gain in straight lines (proportional gain)
VELOCITY_STRAIGHT = 0.7     # Max speed: 0.85 (anything above this makes the robot max out at 0.6m/s in the straights and therefore it cannot make corrections anymore)
GAIN_STRAIGHT = 2.5

# Expert's velocity and steering gain in curves/corners (proportional gain)
VELOCITY_CURVE = 0.25       # Faster: 0.35 (but less reliable/safe)
GAIN_CURVE = 4.15

# Expert's velocity and steering gain during DAgger training (proportional gain)
VELOCITY_DAGGER = 0.8
GAIN_DAGGER = 7.0

# Gain of the D controller (derivative gain) 
D_GAIN = 5.0

FOLLOWING_DISTANCE_CURVE = 0.325    # The following distance in curves/corners
FOLLOWING_DISTANCE_STRAIGHT = 7.0   # The following distance in straight lines
                                    # The expert will project a point ahead of itself (alongside the optimal driving line)
                                    # that is as far from him as the following distance, and it will move towards this point


class PurePursuitExpert:
    def __init__(self, env, ref_velocity=VELOCITY_CURVE, following_distance=FOLLOWING_DISTANCE_CURVE,
                 max_iterations=1000, DAgger=False):
        self.env = env
        self.max_iterations = max_iterations
        self.DAgger = DAgger

        self.following_distance = following_distance
        self.ref_velocity = ref_velocity
        self.P_gain = GAIN_CURVE
        self.D_gain = D_GAIN
        
        self.prev_err = 0

    def predict(self, observation):  # we don't really care about the observation for this implementation

        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)
        
        if closest_point is None:
            return 0.0, 0.0 # Should return done in the environment

        # Look for a point on the ideal driving line, that is as far from the expert as the actual following distance
        # - This point will be: curve_point
        # - We will use curve_point to calculate the steering angle of the expert
        # - The expert will move towards curve_point
        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, tangent = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # This code gets a point on the ideal driving line such that the distance between this point and the agent is 0.3
        # - This point will be: corner_curve_point
        # - We will use corner_curve_point to calculate/predict, whether a corner/curve is coming (after a straight line)
        iterations = 0
        corner_lookup_distance = 0.375
        corner_curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            corner_follow_point = closest_point + closest_tangent * corner_lookup_distance
            corner_curve_point, corner_tangent = self.env.closest_curve_point(corner_follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if corner_curve_point is not None:
                break

            iterations += 1
            corner_lookup_distance *= 0.5

        # Next, we will calculate the abs_corner_value:
        # - abs_corner_value will tell us, whether the expert is in a curve/corner
        # - if abs_corner_value > 0, the expert is in a curve, otherwise the expert is in a straight

        # The way we calculate the abs_corner_value is the following:
        # - We have the following positions: 
        #       closest_point - it is the closest position on the ideal, right driving lane next to the expert
        #       corner_curve_point - it is the position on the ideal, right driving lane that is 0.3 far from the expert 
        # - If we substract these 2, we will get the vector posVec. This vectors 2nd-axis value will always be 0 (Z value)
        #       If we are in a straight, the value of the vector alongside one axis will be 0.3, 
        #           and it will be 0 alongside the other axis. So: (0, 0, 0.3) or (0.3, 0, 0)
        #       If we are in a curve, the vector will have 2 values alongside 2 axes. So (something 0 something)
        # - We cross multiply the corner_tangent with the vector (0, 1, 0) and we get a new vector rightVec.
        #       If we are in a straight, the value of this vector alongside one axis will be something, 
        #           and it will be 0 alongside the other axis. So: (0, 0, something) or (something, 0, 0)
        #           BUT: It is important, that this vector will have a value on the other axis compared to the previous
        #           posVec vector. So if posVec is (0, 0, 0.3), rightVec is (something, 0, 0), and similarly
        #           if posVec is (0.3, 0, 0) then rightVec (0, 0, something)
        #       If we are in a curve, the vector will have 2 values alongside 2 axes. So (something 0 something)
        # - Finally, we dot multiply posVec and rightVec
        #       In a straight, this value will be 0: (0.3, 0, 0) * (0, 0, something) = 0, and similarly in the other case
        #       In a curve, this value will be bigger than 0: (something, 0, something) * (something, 0, something) != 0

        # Calculate the abs_corner_value based on the previous notes
        posVec = corner_curve_point - closest_point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(corner_tangent, upVec)
        curve_value = np.dot(posVec, rightVec)
        abs_corner_value = np.absolute(curve_value)

        # If we are in a straight and a corner/curve is coming, set the steering gain and the velocity for the curve
        if(abs_corner_value > POSITION_THRESHOLD_CURVE) and (self.ref_velocity > VELOCITY_CURVE):
            self.P_gain = GAIN_CURVE
            self.ref_velocity = VELOCITY_CURVE
            self.following_distance = FOLLOWING_DISTANCE_CURVE
        
        # If we are in a curve/corner and a straight is coming, set the steering gain and the velocity for the straight
        if (abs_corner_value < POSITION_THRESHOLD_STRAIGHT) and (self.ref_velocity < VELOCITY_STRAIGHT): 
            self.P_gain = GAIN_STRAIGHT
            self.ref_velocity = VELOCITY_STRAIGHT
            self.following_distance = FOLLOWING_DISTANCE_STRAIGHT

        if self.DAgger:
            self.P_gain = GAIN_DAGGER
        #    self.ref_velocity = VELOCITY_DAGGER

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        # Compute steering angle
        err = np.dot(self.env.get_right_vec(), point_vec)
        derr = err - self.prev_err
        self.prev_err = err
        steering = self.P_gain * -err - self.D_gain * derr

        if self.DAgger:
            action = (self.ref_velocity*0.9), steering
        else:
            action = self.ref_velocity, steering

        return action