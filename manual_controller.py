from controller import Robot
from controller import Keyboard
import struct
import numpy as np
import os

'''
This is a script for controlling robot WASD and saving photos into folder
'''


class PDcontroller:

    def __init__(self, p, d, sampling_period, target=0.0):
        self.target = target
        self.response = 0.0
        self.old_error = 0.0
        self.p = p
        self.d = d
        self.sampling_period = sampling_period

    def process_measurement(self, measurement):
        error = self.target - measurement
        derivative = (error - self.old_error) / self.sampling_period
        self.old_error = error
        self.response = self.p * error + self.d * derivative
        return self.response

    def reset(self):
        self.target = 0.0
        self.response = 0.0
        self.old_error = 0.0


class MotorController:

    def __init__(self, name, robot, pd):
        self.name = name
        self.robot = robot
        self.pd = pd
        self.motor = None
        self.velocity = 0.0

    def enable(self):
        self.motor = self.robot.getDevice(self.name)
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0.0)

    def update(self):
        self.velocity += self.pd.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)

    def set_target(self, target):
        self.pd.target = target

    def emergency_stop(self):
        self.motor.setVelocity(0.0)
        self.pd.reset()
        self.velocity = 0.0


class MotorCommand:

    def __init__(self, left_velocity, right_velocity, emergency_stop=False):
        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.emergency_stop = emergency_stop


class Camera:
    __CHANNELS = 4

    def __init__(self, name, folder=None):
        self.name = name
        self.frame = None
        self.image_byte_size = None
        self.device = None
        self.folder = folder
        self.id = 0

    def enable(self, timestep):
        self.device = robot.getDevice(self.name)
        self.device.enable(timestep)
        self.image_byte_size = self.device.getWidth() * self.device.getHeight() * Camera.__CHANNELS
        if self.folder is not None:
            for filename in os.listdir(self.folder):
                if filename.endswith('.png'):
                    try:
                        id = int(filename.split('.')[0])
                        if id > self.id:
                            self.id = id
                    except Exception:
                        # Exception is only for files we are not looking for
                        pass
            self.id += 1

    def get_frame(self):
        frame = self.device.getImage()
        if frame is None:
            return self.frame
        frame = struct.unpack(f'{self.image_byte_size}B', frame)
        frame = np.array(frame, dtype=np.uint8).reshape(self.device.getHeight(),
                                                        self.device.getWidth(), Camera.__CHANNELS)
        frame = frame[:, :, 0:3]
        self.frame = frame
        return frame

    def show_frame(self, scale=1.0):
        scaled_frame = cv2.resize(self.frame, (0, 0), fx=scale, fy=scale)
        cv2.imshow(self.name, scaled_frame)

    def save_frame(self):
        cv2.imwrite(f'{self.folder}/{self.id}.png', self.frame)
        self.id += 1


CRUISING_SPEED = 5.0
TURN_SPEED = CRUISING_SPEED / 2.0

motor_NET = {
    ord('W'): MotorCommand(CRUISING_SPEED, CRUISING_SPEED),
    ord('S'): MotorCommand(-CRUISING_SPEED, -CRUISING_SPEED),
    ord('A'): MotorCommand(-TURN_SPEED, TURN_SPEED),
    ord('D'): MotorCommand(TURN_SPEED, -TURN_SPEED),
    ord('X'): MotorCommand(0.0, 0.0),
    ord('E'): MotorCommand(0.0, 0.0, True)
}

robot = Robot()
timestep = 64  # int(robot.getBasicTimeStep())
timestep_seconds = timestep / 1000.0

motor_left = MotorController('left wheel', robot, PDcontroller(0.01, 0.0001, timestep_seconds))
motor_right = MotorController('right wheel', robot, PDcontroller(0.01, 0.0001, timestep_seconds))
motor_left.enable()
motor_right.enable()

cv2.startWindowThread()

camera = Camera('kinect color', 'C:\Obrazki')
camera.enable(timestep)

keyboard = Keyboard()
keyboard.enable(timestep)

PHOTO_PERIOD = 1
last_photo_timestamp = 0
timestamp = 0
auto_photo_mode = False

while robot.step(timestep) != -1:
    timestamp += timestep
    camera.get_frame()
    camera.show_frame(scale=3.0)
    motor_left.update()
    motor_right.update()
    key = keyboard.getKey()
    if auto_photo_mode and timestamp - last_photo_timestamp >= PHOTO_PERIOD * 1000:
        camera.save_frame()
        last_photo_timestamp = timestamp
        print(f'Foto! {timestamp // 1000} foto.')
    if key in motor_NET.keys():
        cmd = motor_NET[key]
        if cmd.emergency_stop:
            motor_left.emergency_stop()
            motor_right.emergency_stop()
        else:
            motor_left.set_target(cmd.left_velocity)
            motor_right.set_target(cmd.right_velocity)
    elif key == ord('Q'):
        print('fotka')
        camera.save_frame()
    elif key == ord('O'):
        auto_photo_mode = not auto_photo_mode
    cv2.waitKey(timestep)
