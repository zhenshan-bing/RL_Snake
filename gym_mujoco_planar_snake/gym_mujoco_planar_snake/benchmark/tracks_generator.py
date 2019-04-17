import numpy as np
import math

class TracksGenerator:

    def __init__(self, target_v = 0.3, head_target_dist = 4.0, target_dist_min = 2.0, target_dist_max = 6.0 ):
    #def __init__(self, target_v=0.5, head_target_dist=2.5, target_dist_min=2.5, target_dist_max=2.5):

        self.target_dist_max = target_dist_max  # 7.5  # 8
        self.target_dist_min = target_dist_min
        #self.target_v = 0.15 # -> 0.1
        # self.target_v = 0.2 # -> 0.1
        # self.target_v = 0.3 # -> 0.0135 mean_target_v 0.3597
        self.target_v = target_v

        # 0.1
        self.head_target_dist = head_target_dist




    def calculate_distance(self, head_x, head_y, target_x, target_y):
        #head_pos = np.array([head_x, head_y, 0])
        #target_pos = self.get_body_com('target_ball')
        #target_pos = np.array([target_x, target_y, 0])
        #dist = np.linalg.norm(target_pos - head_pos)
        #dist = np.abs(dist)
        #print(target_x, head_x)
        dist = math.sqrt(((target_x - head_x) ** 2) + ((target_y - head_y) ** 2))
        return dist

    def gen_line_step(self, head_x, head_y, target_x, target_y, dt):
        x = target_x
        y = 0

        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        #print(head_x, target_x)
        #print(head_y, target_y)
        #print(current_dist)

        if current_dist < self.target_dist_min:
            #print('<')
            x = head_x + self.target_dist_min

        elif current_dist > self.target_dist_max:
            #print('>')
            pass
            #x = head_x + self.target_dist_max
            #x = 0 + self.target_dist_min
        #    1+1
        else:
            #print('+')
            x += self.target_v * dt

        return x, y

    def gen_wave_step(self, head_x, head_y, target_x, target_y, dt):
        start_sin_at = 5
        period = 0.2
        amplitude = 5

        x = target_x
        y = target_y
        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)
        # x += self.head_target_dist

        if x >= start_sin_at:
            y = amplitude * np.sin(period * (x - start_sin_at))

        else:
            y = 0

        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        # x = head_x + self.target_dist_max
        else:

            #y = amplitude * np.sin(period * (x - start_sin_at))
            y_diff = np.abs(y - target_y)
            y_way = np.sqrt(y_diff**2+y_diff**2)

            # not the best solution
            x = x + (self.target_v * dt) - 5*y_way*dt

            # x += self.target_v * dt

        return x, y

    def gen_zigzag_step(self, head_x, head_y, target_x, target_y, dt, plot=False):
        c = 10

        # easy
        #d = 0.06666# 0.07
        #start_sin_at = 7.5  # 8

        # 90 deg
        #d = 0.1
        #e = 0

        # 60 deg
        d = 0.056
        e = 4

        start_sin_at = 5  # 8

        x = target_x
        y = target_y

        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        def a():
            return c * (-1 + 2 * math.fmod(math.floor(d * (x+e)), 2))

        def b():
            return - c * math.fmod(math.floor(d * (x+e)), 2)


        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        # x = head_x + self.target_dist_max
        else:

            #x += self.target_v * dt
            if x >= start_sin_at:
                # 90 deg
                #x += (self.target_v * dt) / np.sqrt(2)

                # 60 deg
                x += (self.target_v * dt) *0.872
                y = (d * (x+e) - math.floor(d * (x+e))) * a() + b() + c / 2
            else:
                x += self.target_v * dt
                y = 0


        # hot fix
        if(plot):
            if x >= start_sin_at:
                # 90 deg
                #x += (self.target_v * dt) / np.sqrt(2)

                # 60 deg
                x += (self.target_v * dt) *0.872
                y = (d * (x+e) - math.floor(d * (x+e))) * a() + b() + c / 2
            else:
                x += self.target_v * dt
                y = 0


        return x, y

    def gen_circle_step(self, head_x, head_y, target_x, target_y, dt):
        start_sin_at = 10  # 8
        radius = start_sin_at
        x = target_x
        y = target_y

        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)


        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        # x = head_x + self.target_dist_max
        else:
            x += self.target_v * dt


            if x >= start_sin_at:
                alpha_head = math.degrees(math.atan2(head_y, head_x))

                # angle alpha at a known distance
                # Law of cosines
                # b^2 = a^2 + c^2 - 2ac * cos beta
                #a = self.head_target_dist - 2
                a = current_dist + self.target_v * dt
                b = radius
                c = radius
                cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
                beta = math.degrees(math.acos(cos_alpha))

                alpha_target = alpha_head + beta

                x = math.cos(math.radians(alpha_target)) * radius
                y = math.sin(math.radians(alpha_target)) * radius







        """
        # before circle
        if head_x <= radius - self.head_target_dist:
        #if math.sqrt(head_x ** 2 + head_y ** 2) <= radius - self.head_target_dist:
            x = head_x + self.head_target_dist
            y = 0
            # print('before circle')
        else:
            # a = math.degrees(math.atan(y_head / x_head))
            alpha_head = math.degrees(math.atan2(head_y, head_x))

            # angle alpha at a known distance
            # Law of cosines
            # b^2 = a^2 + c^2 - 2ac * cos beta
            a = self.head_target_dist
            b = radius
            c = radius
            cos_alpha = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
            beta = math.degrees(math.acos(cos_alpha))

            alpha_target = alpha_head + beta

            x = math.cos(math.radians(alpha_target)) * radius
            y = math.sin(math.radians(alpha_target)) * radius
        """

        return x, y



    current_segment_idx = 0
    current_segment_start_x = 0
    current_segment_start_y = 0
    def gen_track_angles(self, angles, target_x, target_y, dist):

        segment_angle = angles[self.current_segment_idx]
        segment_length = 5

        segment_dist = math.sqrt(((self.current_segment_start_x-target_x)**2) + ((self.current_segment_start_y-target_y)**2))
        new_segment_dist = dist + segment_dist

        if new_segment_dist > segment_length:
            self.current_segment_start_x = self.current_segment_start_x + math.cos(math.radians(segment_angle)) * segment_length
            self.current_segment_start_y = self.current_segment_start_y + math.sin(math.radians(segment_angle)) * segment_length



            # add rest length
            if self.current_segment_idx < len(angles)-1:
                self.current_segment_idx += 1

            segment_angle = angles[self.current_segment_idx]

            new_segment_dist = new_segment_dist - segment_length

        x = self.current_segment_start_x + math.cos(math.radians(segment_angle)) * new_segment_dist
        y = self.current_segment_start_y + math.sin(math.radians(segment_angle)) * new_segment_dist

        return x, y


    currentDist = 0
    def gen_random_step(self, head_x, head_y, target_x, target_y, dt, seed=6, ignore_head=True):

        max_degree_target_change = 60
        segments = 15
        segment_length = 3
        angles = [0, -30, -10, 40, 80, 120, 160, 170, 190, 190, 180, 200, 220, 250, 280]

        np.random.seed(seed)

        angles2 = [0]
        for i in range(segments):
            a = np.random.randint(angles2[-1]-max_degree_target_change, angles2[-1]+max_degree_target_change)
            angles2.append(a)



        # do step
        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        if current_dist < self.target_dist_min:
            #print('<')
            #x = head_x + self.target_dist_min
            dist = self.target_dist_min - current_dist + self.target_v * dt

        elif current_dist > self.target_dist_max:
            #print('>')
            dist = 0.0
            #x = head_x + self.target_dist_max
            #x = 0 + self.target_dist_min
        #    1+1
        else:
            #print('+')
            dist = self.target_v * dt


        if ignore_head:
            dist = self.target_v * dt

        x, y = self.gen_track_angles(angles2, target_x, target_y, dist)

        return x, y

