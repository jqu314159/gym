import pygame
import math
import numpy as np

screen_width = 3828
screen_height = 1596
check_point = ((3300, 1500), (3472, 1180), (3715, 650),
  (3500, 130), (3215, 335), (3307, 625), (3356, 950),
  (3330, 1120), (2910, 1230), (3000, 750), (2750, 700),
  (2600, 800), (2250, 340), (1450, 230), (2570, 580),
  (2400, 1100), (1550, 900), (320, 1190), (2000,1500)
)

class Reward_rate:
    def __init__(self):
        """
          Set about 50 times iteration to reach the relay point
          Set about 4096 times iteration
          Set about 5 times iteration in the pass_over area
        """
        self.arrive_relay_point = 60
        self.arrive_big_relay_point = 180
        self.pass_over_relay_point = -0
        self.keep_away_relay_point = -30
        self.arrive_goal_point = 300
        self.collision_car = -300
        self.collision_car_base = -40
        self.encourage_angular_acceleration = 4
        self.encourage_acceleration = 8
        self.encourage_car_speed = 10
        self.encourage_more_distance = 0.002
        self.touch_safe_distance = -10
        self.encourage_time_cost = 0
        self.conside_step = False
        

        
class Car:
    def __init__(self, car_file, map_file, pos):
        self.screen_width = 3828
        self.screen_height = 1596
        
        self.surface = pygame.image.load(car_file)
        self.map = pygame.image.load(map_file)
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = pos
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.current_check = 0
        self.prev_distance = 0
        self.cur_distance = 0
        self.goal = False
        self.check_flag = False
        self.distance = 0
        self.time_spent = 0
        self.old_distance = 5000
        self.last_distance = 5000

        self.four_points = [[[],[]],[[],[]],[[],[]],[[],[]]]
        self.safe_distance_point = [[[],[]],[[],[]],[[],[]],[[],[]]]
        self.max_car_angular = 900.0
        self.max_car_speed = 10.0
        self.min_car_speed = -1.0
        self.angular_acceleration = 0
        self.acceleration_c = 0
        self.min_angular_acceleration_action = -4
        self.max_angular_acceleration_action = 4
        self.min_acceleration_action = -4
        self.max_acceleration_action = 4
        self.current_speed = 0
        self.current_angle = 0
        
        for d in range(-90, -29, 30):
            self.check_radar(d)
        self.check_radar(-5)
        self.check_radar(0)
        self.check_radar(5)
        for d in range(30, 91, 30):
            self.check_radar(d)

        for d in range(-90, -29, 30):
            self.check_radar_for_draw(d)
        self.check_radar_for_draw(-5)
        self.check_radar_for_draw(0)
        self.check_radar_for_draw(5)
        for d in range(30, 91, 30):
            self.check_radar_for_draw(d)

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)

    def draw_collision(self, screen):
        for i in range(4):
            x = self.four_points[i][0]
            y = self.four_points[i][1]
            safe_x = self.safe_distance_point[i][0]
            safe_y = self.safe_distance_point[i][1]
            if(y==[]):
                x = 2070
                y = 1520
                safe_x = x
                safe_y = y
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 5)
            pygame.draw.circle(screen, (50, 50, 50), (safe_x, safe_y), 5)

    def draw_radar(self, screen):
        for r in self.radars_for_draw:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self):
        self.is_alive = True
        i = 0
        for p in self.four_points:
            i += 1
            if self.map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break


    def check_radar(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.current_angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.current_angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.current_angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.current_angle + degree))) * len)

        dist = math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        self.radars.append([(x, y), dist])


    def check_radar_for_draw(self, degree):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.current_angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.current_angle + degree))) * len)

        while not self.map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.current_angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.current_angle + degree))) * len)

        dist = math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        self.radars_for_draw.append([(x, y), dist])

    def check_checkpoint(self):
        p = check_point[self.current_check]
        self.prev_distance = self.cur_distance
        dist = get_distance(p, self.center)
        if dist < 70:
            self.current_check += 1
            self.prev_distance = 9999
            self.check_flag = True
            if self.current_check >= len(check_point):
                self.current_check = 0
                self.goal = True
            else:
                self.goal = False
        self.cur_distance = dist
        
    def acceleration_angle(self, action):
        angular_acceleration = action[0]
        angular_acceleration = min(self.max_angular_acceleration_action, max(angular_acceleration, self.min_angular_acceleration_action))
        
        self.angular_acceleration = angular_acceleration
        self.current_angle += angular_acceleration
    	
    def acceleration(self, action):
        acceleration = action[1]
        acceleration = min(self.max_acceleration_action, max(acceleration, self.min_acceleration_action)) 
        self.acceleration_c = acceleration
        self.current_speed += acceleration
        self.current_speed = min(self.max_car_speed, max(self.current_speed, self.min_car_speed))
    
    def update(self):

        self.rotate_surface = rot_center(self.surface, self.current_angle)
        self.distance += self.current_speed
        self.time_spent += 1
        new_dif_x = math.cos(math.radians(360 - self.current_angle)) * self.current_speed
        new_dif_y = math.sin(math.radians(360 - self.current_angle)) * self.current_speed
        
        self.pos[0] = max(20, min(screen_width - 120, self.pos[0] + new_dif_x))
        self.pos[1] = max(20, min(screen_height - 120, self.pos[1] + new_dif_y))

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.current_angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.current_angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.current_angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.current_angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.current_angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.current_angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.current_angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.current_angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]
        len = 50
        safe_distance_left_top = [
          self.center[0] + math.cos(math.radians(360 - (self.current_angle + 30))) * len,
          self.center[1] + math.sin(math.radians(360 - (self.current_angle + 30))) * len
        ]
        safe_distance_right_top = [
          self.center[0] + math.cos(math.radians(360 - (self.current_angle + 150))) * len,
          self.center[1] + math.sin(math.radians(360 - (self.current_angle + 150))) * len
        ]
        safe_distance_left_bottom = [
          self.center[0] + math.cos(math.radians(360 - (self.current_angle + 210))) * len,
          self.center[1] + math.sin(math.radians(360 - (self.current_angle + 210))) * len
        ]
        safe_distance_right_bottom = [
          self.center[0] + math.cos(math.radians(360 - (self.current_angle + 330))) * len,
          self.center[1] + math.sin(math.radians(360 - (self.current_angle + 330))) * len
        ]

        self.safe_distance_point = [safe_distance_left_top,safe_distance_right_top, safe_distance_left_bottom, safe_distance_right_bottom]

class PyGame2D:
    def __init__(self):
        pygame.init()
        #self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.car = Car('custom_data/car.png', 'custom_data/map2.png', [2070, 1520])
        self.reward_rate = Reward_rate()
        self.game_speed = 60
        self.mode = 0
        self.total_step = 5000
        self.current_step = 0
        self.max_iteration = 3000
        self.current_iteration = 0
        

        self.pos_pass1 = []
        self.pos_pass2 = []
        self.pos_pass3 = []
        self.pos_pass4 = []
        self.pos_pass5 = []
        self.pos_pass6 = []
        self.pos_pass7 = []
        self.pos_pass8 = []
        self.pos_pass9 = []
        self.angle_pass1 = 0
        self.angle_pass2 = 0
        self.angle_pass3 = 0
        self.angle_pass4 = 0
        self.angle_pass5 = 0
        self.angle_pass6 = 0
        self.angle_pass7 = 0
        self.angle_pass8 = 0
        self.angle_pass9 = 0
        self.Hit_wall_times = 0
        
    def replay_back_iteration(self):
        if self.car.is_alive == False: 
            if self.pos_pass9 == []:
                self.car.pos = [2070 - 50, 1520 -50]
                self.car.current_angle = 0
            else:
                self.car.pos = self.pos_pass9.copy()
                #self.car.current_angle = self.angle_pass9 + np.random.uniform(self.car.min_angular_acceleration_action, self.car.max_angular_acceleration_action, size=1)
                self.car.current_angle = self.angle_pass9
                self.pos_pass8 = self.pos_pass9.copy()
                self.pos_pass7 = self.pos_pass9.copy()
                self.pos_pass6 = self.pos_pass9.copy()
                self.pos_pass5 = self.pos_pass9.copy()
                self.pos_pass4 = self.pos_pass9.copy()
                self.pos_pass3 = self.pos_pass9.copy()
                self.pos_pass2 = self.pos_pass9.copy()
                self.pos_pass1 = self.pos_pass9.copy()
                self.angle_pass8 = self.angle_pass9
                self.angle_pass7 = self.angle_pass9
                self.angle_pass6 = self.angle_pass9
                self.angle_pass5 = self.angle_pass9
                self.angle_pass4 = self.angle_pass9
                self.angle_pass3 = self.angle_pass9
                self.angle_pass2 = self.angle_pass9
                self.angle_pass1 = self.angle_pass9
                self.Hit_wall_times += 1
                self.car.current_speed = self.car.min_car_speed
        elif self.pos_pass9 == self.pos_pass8:
            self.pos_pass9 = self.pos_pass8.copy()
            self.pos_pass8 = self.pos_pass7.copy()
            self.pos_pass7 = self.pos_pass6.copy()
            self.pos_pass6 = self.pos_pass5.copy()
            self.pos_pass5 = self.pos_pass4.copy()
            self.pos_pass4 = self.pos_pass3.copy()
            self.pos_pass3 = self.pos_pass2.copy()
            self.pos_pass2 = self.pos_pass1.copy()
            self.pos_pass1 = self.car.pos.copy()
            self.angle_pass9 = self.angle_pass8
            self.angle_pass8 = self.angle_pass7
            self.angle_pass7 = self.angle_pass6
            self.angle_pass6 = self.angle_pass5
            self.angle_pass5 = self.angle_pass4
            self.angle_pass4 = self.angle_pass3
            self.angle_pass3 = self.angle_pass2
            self.angle_pass2 = self.angle_pass1
            self.angle_pass1 = self.car.current_angle
        else:
            self.pos_pass9 = self.pos_pass8.copy()
            self.pos_pass8 = self.pos_pass7.copy()
            self.pos_pass7 = self.pos_pass6.copy()
            self.pos_pass6 = self.pos_pass5.copy()
            self.pos_pass5 = self.pos_pass4.copy()
            self.pos_pass4 = self.pos_pass3.copy()
            self.pos_pass3 = self.pos_pass2.copy()
            self.pos_pass2 = self.pos_pass1.copy()
            self.pos_pass1 = self.car.pos.copy()
            self.angle_pass9 = self.angle_pass8
            self.angle_pass8 = self.angle_pass7
            self.angle_pass7 = self.angle_pass6
            self.angle_pass6 = self.angle_pass5
            self.angle_pass5 = self.angle_pass4
            self.angle_pass4 = self.angle_pass3
            self.angle_pass3 = self.angle_pass2
            self.angle_pass2 = self.angle_pass1
            self.angle_pass1 = self.car.current_angle
            self.Hit_wall_times = 0
        
    def action(self, action):
        self.car.acceleration_angle(action)
        self.car.acceleration(action)
        
        self.car.update()
        self.car.check_collision()
        self.replay_back_iteration()
        self.car.check_checkpoint()

        self.car.radars.clear()
        for d in range(-90, -29, 30):
            self.car.check_radar(d)
        self.car.check_radar(-5)
        self.car.check_radar(0)
        self.car.check_radar(5)
        for d in range(30, 91, 30):
            self.car.check_radar(d)

    def evaluate(self):
        
        relay_point_distance = math.sqrt(math.pow(self.car.center[0]- check_point[self.car.current_check][0], 2) + math.pow(self.car.center[1]- check_point[self.car.current_check][1], 2))
       
        reward = 0
        reward += self.reward_rate.encourage_time_cost
        Conside_steps = 1
        if self.reward_rate.conside_step == True:
           Conside_steps =  (self.total_step-self.current_step) / self.total_step
        if self.car.check_flag:
            self.car.check_flag = False
            reward += self.reward_rate.arrive_relay_point
            self.car.time_spent = 0
            self.car.old_distance = 5000
            
            if self.car.current_check > 11:
                reward += self.reward_rate.arrive_big_relay_point
        
        if self.car.old_distance < relay_point_distance:

            if relay_point_distance < 350 and self.car.last_distance < relay_point_distance:
                reward += self.reward_rate.pass_over_relay_point
            elif self.car.last_distance < relay_point_distance:
                reward += self.reward_rate.keep_away_relay_point
        else:
            self.car.old_distance = relay_point_distance
            
        self.car.last_distance = relay_point_distance
        
        if not self.car.is_alive:
            reward += self.reward_rate.collision_car_base + self.reward_rate.collision_car * (1- Conside_steps)
        elif self.car.goal:
            reward += self.reward_rate.arrive_goal_point
            reward += self.car.distance * self.reward_rate.encourage_more_distance
            
        reward += 2 * self.reward_rate.encourage_angular_acceleration * (abs(self.car.angular_acceleration) / self.car.max_angular_acceleration_action) * (Conside_steps -0.5)
        
        reward += 2 * self.reward_rate.encourage_acceleration  * (abs(self.car.angular_acceleration) /self.car.max_acceleration_action) * (Conside_steps -0.5)
        
        reward += self.reward_rate.encourage_car_speed * self.car.current_speed / self.car.max_car_speed
        
        for p in self.car.safe_distance_point:
            if self.car.map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                reward += self.reward_rate.touch_safe_distance
        return reward
        
    def  double_reward_evaluate(self):
    
        relay_point_distance = math.sqrt(math.pow(self.car.center[0]- check_point[self.car.current_check][0], 2) + math.pow(self.car.center[1]- check_point[self.car.current_check][1], 2))
        reward = 0
        reward2 = 0
        reward += self.reward_rate.encourage_time_cost
        Conside_steps = 1
        if self.reward_rate.conside_step == True:
           Conside_steps =  (self.total_step-self.current_step) / self.total_step
        
        if self.car.check_flag:
            self.car.check_flag = False
            reward2 += self.reward_rate.arrive_relay_point
            self.car.time_spent = 0
            self.car.old_distance = 5000
            if self.car.current_check > 11:
                reward2 += self.reward_rate.arrive_big_relay_point
            #reward2 += self.car.distance * self.reward_rate.encourage_more_distance
            self.car.distance = 0 
        
        if self.car.old_distance < relay_point_distance:

            if relay_point_distance < 350 and self.car.last_distance < relay_point_distance:
                reward += self.reward_rate.pass_over_relay_point
            elif self.car.last_distance < relay_point_distance:
                reward += self.reward_rate.keep_away_relay_point
        else:
            self.car.old_distance = relay_point_distance
            
        self.car.last_distance = relay_point_distance
        
        if not self.car.is_alive:
            reward += self.reward_rate.collision_car_base + self.reward_rate.collision_car * (1- Conside_steps)
            
        elif self.car.goal:
            reward2 += self.reward_rate.arrive_goal_point
            reward2 += self.car.distance * self.reward_rate.encourage_more_distance
            
        reward += 2 * self.reward_rate.encourage_angular_acceleration * (abs(self.car.angular_acceleration) / self.car.max_angular_acceleration_action) * (Conside_steps -0.5)
        
        reward += 2 * self.reward_rate.encourage_acceleration  * (abs(self.car.angular_acceleration) /self.car.max_acceleration_action) * (Conside_steps -0.5)
        
        reward += self.reward_rate.encourage_car_speed * self.car.current_speed / self.car.max_car_speed
        
        for p in self.car.safe_distance_point:
            if self.car.map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                reward += self.reward_rate.touch_safe_distance
        if self.current_iteration >= self.max_iteration-1:
            reward2 += min(self.car.distance * self.reward_rate.encourage_more_distance, self.reward_rate.arrive_relay_point/ 2)
        #print("reward=", reward)
        return reward ,reward2

    def is_done(self):
        
        if self.current_iteration >= self.max_iteration or self.car.goal:
            self.car.current_check = 0
            self.car.distance = 0
            self.current_iteration = 0
            
            return True
        else:
            self.current_iteration += 1
            return False

    def observe(self):
        # return state
        radars = self.car.radars
        ret = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i, r in enumerate(radars):
        	
            ret[i] = r[1] / 30
        ret[9] = float(self.car.center[0]) / 100.0
        ret[10] = float(self.car.center[1])/ 100.0
        ret[11] = float(check_point[self.car.current_check][0]) / 100.0
        ret[12] = float(check_point[self.car.current_check][1]) / 100.0
        ret[13] = math.cos(math.radians(self.car.current_angle ))
        ret[14] = math.sin(math.radians(self.car.current_angle ))
        return np.asarray(ret)

    def view(self):
        # draw game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode += 1
                    self.mode = self.mode % 3

        self.screen.blit(self.car.map, (0, 0))
        if self.mode == 1:
            self.screen.fill((0, 0, 0))

        self.car.radars_for_draw.clear()
        for d in range(-90, -29, 30):
            self.car.check_radar_for_draw(d)
        self.car.check_radar_for_draw(-5)
        self.car.check_radar_for_draw(0)
        self.car.check_radar_for_draw(5)
        for d in range(30, 91, 30):
            self.car.check_radar_for_draw(d)

        self.car.draw_collision(self.screen)
        self.car.draw_radar(self.screen)
        self.car.draw(self.screen)
        #print(self.car.current_check)
        pygame.draw.circle(self.screen, (255, 255, 0), check_point[self.car.current_check], 70, 1)
        pygame.display.flip()
        self.clock.tick(self.game_speed)


def get_distance(p1, p2):
	return math.sqrt(math.pow((p1[0] - p2[0]), 2) + math.pow((p1[1] - p2[1]), 2))

def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image
