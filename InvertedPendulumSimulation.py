# -*- coding: utf-8 -*-

#***********************************************************
#Inverted Pendulum without rotary encoder
#2021/07/04 Hideo Miyauchi
#***********************************************************

#***********************************************************
#Inverted Pendulum
#Calculate the optimal feedback gain
#Simulate the motion
#2019/03/01 N. Beppu
#***********************************************************
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
#from matplotlib.animation import PillowWriter

#===========================================================
#Model parameters
#===========================================================
#-------------------------------------------------
#Tamiya sports tire set
#-------------------------------------------------
#The radius of the wheel (m)
r_wheel = 0.028

#-------------------------------------------------
#Tamiya high power gear box HE
#-------------------------------------------------
#The gear ratio
gear_ratio = 64.8   #high power gear box

#-------------------------------------------------
#Motor (RE-260RA-2670)
#-------------------------------------------------
#resistance
Rm = 2.4
#The back electromotive force constant (V.s/rad)
kb = 0.0024

#-------------------------------------------------
#parameter summary
#-------------------------------------------------
print("******************************")
print("*        Parameters          *")
print("******************************")
print("Wheel Parameters")
print("r_wheel = " + str(r_wheel) + " (m)")

print("------------------------------")
print("Motor Parameters")
print("kb = " + str(kb) + " (V.s/rad)")
print("Rm = " + str(Rm) + " (Ohm)")
print("Gear Ratio = " + str(gear_ratio) + "\n")
print("")

#===========================================================
#Calculate matrix A, B, C (continuous time)
#===========================================================
#-------------------------------------------------
#matrix A (continuous time)
#-------------------------------------------------
A = np.array([
    [  0,           1,           0,           0,        ],
    [ 44.2210467,   0,           0,           0.95912734],
    [  0,           0,           0,           1,        ],
    [-22.59588735,  0,           0,          -5.98850358]
])

#-------------------------------------------------
#matrix B (continuous time)
#-------------------------------------------------
B = np.array([
    [ 0         ],
    [-6.16722829],
    [ 0         ],
    [38.50632446]
])

#-------------------------------------------------
#matrix summary (continuous time)
#-------------------------------------------------
print("******************************")
print("*  Matrix (continous time)   *")
print("******************************")
print("Matrix A (continuous time)")
print(A)
print("")

print("------------------------------")
print("Matrix B (continuous time)")
print(B)
print("")

#===========================================================
#Calculate the optimal feedback gain
#Quadratic cost function: x^TQx + u^TRu
#discrete time Riccati equation: A^TPA-P-A^TPB(B^TPB+R)^-1B^TPA+Q=0
#===========================================================
print("Gain (calculated)")
Gain = np.array([[29.87522919, 4.59857246, 0.09293, 0.37006248]])
print(Gain)
print("")

#===========================================================
#Simulation
#===========================================================
#initial value
theta_0 = 20 #degree
x = np.array( [[theta_0 * math.pi / 180], [0], [0], [0]] )

#total number of the step
num = 1000

#variables
time = []
theta_array = []
theta_array2 = []
theta_dot_array = []
theta_dot_array2 = []
V_array = []
I_array = []

#initialize the lists
time.append(0)
theta_array.append(x[0][0]*180/math.pi) # degree
theta_dot_array.append(x[1][0]*180/math.pi) # degree/sec
theta_array2.append(x[2][0]*r_wheel*100) # cm
theta_dot_array2.append(x[3][0]*r_wheel*100) # cm/sec

#calculate the initial value of motor voltage
Vlimit = 10
Vin = np.dot(Gain, x)[0][0]
if Vin > Vlimit:
    Vin = Vlimit
if Vin < -Vlimit:
    Vin = -Vlimit
V_array.append(Vin)

#calculate the initial value of motor current
I_array.append( (Vin - kb*x[3][0]*gear_ratio)/Rm )

#differential equation (continuous time)
def diff_equation(t, x, vin):
    return np.dot(A, x) + B * vin

#Runge-Kutta method
def rungeKuttaSolver(t, x, dt, vin):
    j1 = diff_equation(t, x, vin) * dt
    j2 = diff_equation(t + dt / 2.0, x + j1 / 2.0, vin) * dt
    j3 = diff_equation(t + dt / 2.0, x + j2 / 2.0, vin) * dt
    j4 = diff_equation(t + dt, x + j3, vin) * dt
    return x + (j1 + j2 * 2.0 + j3 * 2.0 + j4) / 6.0

#sampling rate of the discrete time system
T = 0.01 #sec

#sensor characteristic
driftGyro = 0.1 # drift of gyro sensor
scaleGyro = 0.2 # standard deviation of gyro sensor
driftAccel = 0.1 # drift of accelerometer
scaleAccel = 0.2 # standard deviation of accelerometer
offsetDistance = 42 # distance equivalent to drift

#variables for movement
targetSpeed = 0
deltaSpeed = 0
permitMove = False

#variables for the now or before state
beforeGyroOmega = 0
nowGyroOmega = 0
beforeAccelTheta = x[0][0]
nowAccelTheta = 0
beforeTheta = x[0][0]
nowTheta = 0
beforeDelayedVin = 0
nowDelayedVin = 0
beforeVin = 0
beforeSpeed = 0
nowSpeed = 0
beforeDistance = 0
nowDistance = 0

#calculation loop
for i in range(num-1):

    #calculate the next state
    x = rungeKuttaSolver(i, x, T, Vin)

    #angles, angular rates
    theta_array.append(x[0][0] * 180/math.pi) #degree
    theta_dot_array.append(x[1][0] * 180/math.pi) #degree/s
    theta_array2.append(x[2][0] * r_wheel * 100) #cm
    theta_dot_array2.append(x[3][0] * r_wheel * 100) #cm/s

    #simulate gyro sensor
    rawGyro = np.random.normal(x[1][0] + driftGyro, scale=scaleGyro)
    k = 0.3
    nowGyroOmega = (1 - k) * beforeGyroOmega + k * rawGyro

    #simulate accelerometer
    rawAccel = np.random.normal(x[0][0] + driftAccel, scale=scaleAccel)
    k = 0.3
    nowAccelTheta = (1 - k) * beforeAccelTheta + k * rawAccel

    #calculate the angle using complementary filter
    k = 0.06
    nowTheta = (1 - k) * (beforeTheta + nowGyroOmega * T) + k * nowAccelTheta

    #speed is proportional to the delayed motor voltage
    k = 0.05
    nowDelayedVin = (1 - k) * beforeDelayedVin + k * beforeVin
    nowSpeed = nowDelayedVin * 8 # level matching

    #calculate the distance by integrating the velocity
    nowDistance = beforeDistance + (nowSpeed + beforeSpeed) * T / 2

    # move to the right direction
    if (i == 200):
        targetSpeed = 20
        deltaSpeed = 0
        permitMove = True

    # stop moving to the right direction
    if (i == 400):
        targetSpeed = 0
        permitMove = True

    # move to the left direction
    if (i == 600):
        targetSpeed = -20
        deltaSpeed = 0
        permitMove = True

    # stop moving to the left direction
    if (i == 800):
        targetSpeed = 0
        permitMove = True

    # movement processing
    if (permitMove == True):
        nowDistance = 0
        if (targetSpeed > 0): #
            if (deltaSpeed < targetSpeed):
                deltaSpeed = deltaSpeed + 0.3
                if (deltaSpeed > targetSpeed):
                    deltaSpeed = targetSpeed
        elif (targetSpeed < 0): #
            if (deltaSpeed > targetSpeed):
                deltaSpeed = deltaSpeed - 0.3
                if (deltaSpeed < targetSpeed):
                    deltaSpeed = targetSpeed
        elif (deltaSpeed != 0):
            if (deltaSpeed > 0):
                deltaSpeed = deltaSpeed - 0.3
                if (deltaSpeed < 0):
                    deltaSpeed = 0
            elif (deltaSpeed < 0):
                deltaSpeed = deltaSpeed + 0.3
                if (deltaSpeed > 0):
                    deltaSpeed = 0
        else:
            if (abs(nowSpeed) < 1):
                permitMove = False

    #motor voltage
    Vin = np.dot(Gain, np.array([
        [nowTheta],[nowGyroOmega],[nowDistance - offsetDistance],[nowSpeed - deltaSpeed]
    ]))[0][0]
    if Vin > Vlimit:
        Vin = Vlimit
    if Vin < -Vlimit:
        Vin = -Vlimit
    V_array.append( Vin )

    #motor current
    I_array.append( (Vin - kb*x[3][0]*gear_ratio)/Rm )

    #remember current state
    beforeGyroOmega = nowGyroOmega
    beforeAccelTheta = nowAccelTheta
    beforeTheta = nowTheta
    beforeDelayedVin = nowDelayedVin
    beforeVin = Vin
    beforeSpeed = nowSpeed
    beforeDistance = nowDistance

    #time
    time.append( T*(i+1) )

#===========================================================
#Draw graph
#===========================================================
#create figure object
fig = plt.figure( figsize=(8,8) )

#use "subplot" to divide the graph area
ax1 = fig.add_subplot(6,1,1)
ax2 = fig.add_subplot(6,1,2)
ax3 = fig.add_subplot(6,1,3)
ax4 = fig.add_subplot(6,1,4)
ax5 = fig.add_subplot(6,1,5)
ax6 = fig.add_subplot(6,1,6)

#range of the x axis
stop_time = T * num

#angle (degree)
ax1.plot(time,theta_array, color="blue", lw=2)
ax1.set_xlim([0, stop_time])
ax1.set_ylabel("theta_p (deg)")

#angular rate (degree/sec)
ax2.plot(time,theta_dot_array, color="blue", lw=2)
ax2.set_xlim([0, stop_time])
ax2.set_ylabel("dtheta_p (deg/s)")

#position (cm)
ax3.plot(time,theta_array2, color="green", lw=2)
ax3.set_xlim([0, stop_time])
ax3.set_ylabel("theta_w (cm)")

#speed (cm/sec)
ax4.plot(time,theta_dot_array2, color="green", lw=2)
ax4.set_xlim([0, stop_time])
ax4.set_ylabel("dtheta_w (cm/s)")

#voltage (V)
ax5.plot(time,V_array, color="red", lw=2)
ax5.set_xlim([0, stop_time])
ax5.set_ylabel("Voltage (V)")

#current (A)
ax6.plot(time,I_array, color="red", lw=2)
ax6.set_xlim([0, stop_time])
ax6.set_ylabel("Current (A)")
ax6.set_xlabel("time (sec)")

#show the graph
plt.tight_layout()
#plt.show()
#fig.savefig("figure.jpg")

#===========================================================
# Animation
#===========================================================
angle_history =theta_array
x_history =theta_array2

fig2 = plt.figure( figsize=(4,4) )

ratio=2
ax = fig2.add_subplot(111, aspect=ratio, autoscale_on=False, xlim=(-50, 200), ylim=(-10, 30))
ax.grid()
line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, 'aaaaaa', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    line.set_data(
        [ x_history[i], x_history[i] + 16 * ratio * math.sin(angle_history[i] * math.pi / 180) ],
        [ 0, 16 * math.cos(angle_history[i] * math.pi / 180) ]
    )
    time_text.set_text('time = {0:.1f}'.format(i))
    return line, time_text

ani = animation.FuncAnimation(fig2, animate, frames=range(len(x_history)),
    interval=10, blit=False, init_func=init, repeat=False, repeat_delay=5000)
plt.show()
#ani.save("animation.gif", writer="pillow", fps=40)
